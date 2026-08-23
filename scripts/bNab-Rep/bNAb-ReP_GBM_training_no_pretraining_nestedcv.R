library(data.table)
library(h2o)
library(ROCR)
library(geosphere)

start.main <- proc.time()
 

args <- commandArgs(trailingOnly = TRUE)
n.cores <- as.numeric(args[1])
memory  <- as.numeric(args[2])
path    <- args[3]
file.training.data <- args[4]

run.id          <- format(Sys.time(), "%Y%m%d_%H%M%S")
grid.id         <- paste0("grid_", run.id)
model.id        <- "final"
model.cutoff.id <- "final_cutoff"
 

# H2O init
localH2O <- h2o.init(max_mem_size = paste0(memory, "g"),
                     nthreads = n.cores,
                     enable_assertions = FALSE,
                     ip = "localhost", port = 54478)
if (!h2o.clusterIsUp()) stop("H2O cluster is not up.")
 

# Load data
data.hex <- h2o.importFile(path = file.training.data,
                           destination_frame = "MC.hex")
colnames(data.hex)[ncol(data.hex)] <- "response"
train.names <- colnames(data.hex)[1:(ncol(data.hex) - 1)]
data.hex$response <- as.factor(data.hex$response)
 

# Hyperparameter grid
hyper_parameters <- list(
  ntrees          = list(1000),
  max_depth       = list(1, 2, 3, 4, 5, 6),
  learn_rate      = list(0.001, 0.01, 0.05, 0.1, 0.2),
  col_sample_rate = list(sqrt(length(train.names)) / length(train.names),
                         0.1, 0.2, 0.3)
)
 

# NESTED 5x5 CROSS-VALIDATION
 
n.outer <- 5
n.inner <- 5
set.seed(42)
 
# Convert to data frame for fold assignment
df.data <- as.data.frame(data.hex)
n <- nrow(df.data)
 
# Stratified outer fold assignment
response.vals <- as.character(df.data$response)
classes <- unique(response.vals)
 
outer.fold.id <- rep(NA, n)
for (cls in classes) {
  idx.cls <- which(response.vals == cls)
  idx.cls <- sample(idx.cls)
  outer.fold.id[idx.cls] <- ((seq_along(idx.cls) - 1) %% n.outer) + 1
}
 
all.probs <- c()
all.true  <- c()
df.fold.metrics <- data.frame()
 
for (k in 1:n.outer) {
 
  
  cat(sprintf("OUTER FOLD %d of %d\n", k, n.outer))
 
  # Split outer train / test
  idx.outer.train <- which(outer.fold.id != k)
  idx.outer.test  <- which(outer.fold.id == k)
 
  df.outer.train <- df.data[idx.outer.train, ]
  df.outer.test  <- df.data[idx.outer.test,  ]
 
  hex.outer.train <- as.h2o(df.outer.train)
  hex.outer.test  <- as.h2o(df.outer.test)
  hex.outer.train$response <- as.factor(hex.outer.train$response)
  hex.outer.test$response  <- as.factor(hex.outer.test$response)
 
  # Inner grid search
  grid.id.k <- paste0(grid.id, "_outer", k)
 
  gbm.grid.k <- h2o.grid(
    "gbm",
    hyper_params           = hyper_parameters,
    x                      = train.names,
    y                      = "response",
    training_frame         = hex.outer.train,
    nfolds                 = n.inner,
    fold_assignment        = "Stratified",
    grid_id                = grid.id.k,
    balance_classes        = TRUE,
    max_after_balance_size = 5,
    stopping_metric        = "AUC",
    stopping_rounds        = 3,
    stopping_tolerance     = 0.001,
    seed                   = 42
  )
 
  # Best hyperparameters from inner CV
  grid.k        <- h2o.getGrid(grid.id.k, sort_by = "auc", decreasing = TRUE)
  best.params.k <- grid.k@summary_table[1, ]
 
  var.ntrees.k          <- as.numeric(best.params.k$ntrees)
  var.max_depth.k       <- as.numeric(best.params.k$max_depth)
  var.learn_rate.k      <- as.numeric(best.params.k$learn_rate)
  var.col_sample_rate.k <- as.numeric(best.params.k$col_sample_rate)
 
  cat(sprintf("Best inner-CV params: ntrees=%d, depth=%d, lr=%.4f, col_sample=%.4f\n",
              var.ntrees.k, var.max_depth.k, var.learn_rate.k, var.col_sample_rate.k))
 
  # Retrain on full outer-train
  model.id.k <- paste0("final_outer_", k)
 
  gbm.model.k <- h2o.gbm(
    x                      = train.names,
    y                      = "response",
    training_frame         = hex.outer.train,
    model_id               = model.id.k,
    balance_classes        = TRUE,
    max_after_balance_size = 5,
    stopping_metric        = "AUC",
    stopping_rounds        = 3,
    stopping_tolerance     = 0.001,
    ntrees                 = var.ntrees.k,
    max_depth              = var.max_depth.k,
    learn_rate             = var.learn_rate.k,
    col_sample_rate        = var.col_sample_rate.k
  )
 
  # Predict on held-out outer-test fold
  preds.k <- h2o.predict(gbm.model.k, hex.outer.test)
  probs.k <- as.numeric(as.vector(preds.k$p1))
  true.k  <- as.numeric(as.character(df.outer.test$response))
 
  # Accumulate for aggregated metrics
  all.probs <- c(all.probs, probs.k)
  all.true  <- c(all.true,  true.k)
 
  ## Per-fold metrics
  # AUC 
  pred.obj.k  <- prediction(probs.k, true.k)
  auc.k       <- performance(pred.obj.k, "auc")@y.values[[1]]
 
  # TNR vs TPR curve
  perf.k      <- performance(pred.obj.k, "tpr", "tnr")
  vec.tnr.k   <- unlist(perf.k@x.values)
  vec.tpr.k   <- unlist(perf.k@y.values)
  vec.cutoff.k <- unlist(perf.k@alpha.values)
 
  # Cutoff 1: TNR >= 0.9
  cutoff.tnr       <- 0.9
  vec.dist.k       <- abs(vec.tnr.k - cutoff.tnr)
  ind.cutoff.tnr.k <- tail(which(vec.dist.k == min(vec.dist.k)), 1)
  fold.cutoff.k    <- vec.cutoff.k[ind.cutoff.tnr.k]
 
  # Cutoff 2: balanced 
  mat.bisectrix.k       <- cbind(seq(0, 1, 0.001), seq(0, 1, 0.001))
  mat.points.k          <- cbind(vec.tnr.k, vec.tpr.k)
  mat.dist.bisectrix.k  <- dist2Line(mat.points.k, mat.bisectrix.k)
  mat.dist.bis.sorted.k <- mat.dist.bisectrix.k[order(mat.dist.bisectrix.k[, 1]), ]
 
  df.tnr.tpr.k <- data.frame(tnr    = vec.tnr.k,
                              tpr    = vec.tpr.k,
                              cutoff = vec.cutoff.k)
 
  vec.dist.tnr.k <- abs(df.tnr.tpr.k$tnr - as.numeric(mat.dist.bis.sorted.k[1, 2]))
  vec.dist.tpr.k <- abs(df.tnr.tpr.k$tpr - as.numeric(mat.dist.bis.sorted.k[1, 3]))
 
  fold.cutoff.balanced.k <- df.tnr.tpr.k[
    df.tnr.tpr.k$tnr == unique(df.tnr.tpr.k$tnr[which(vec.dist.tnr.k == min(vec.dist.tnr.k))]) &
    df.tnr.tpr.k$tpr == unique(df.tnr.tpr.k$tpr[which(vec.dist.tpr.k == min(vec.dist.tpr.k))]),
    "cutoff"]
 
  fold.cutoff.balanced.k <- fold.cutoff.balanced.k[1]
 
  # if cutoff is NA, all threshold-dependent metrics are NA_real_
  if (is.na(fold.cutoff.balanced.k)) {
 
    acc.k <- NA_real_
    mcc.k <- NA_real_
    f1.k  <- NA_real_
 
    cat(sprintf("Outer fold %d — AUC: %.4f  ACC: NA  MCC: NA  F1: NA  cutoff(balanced): NA\n",
                k, auc.k))
 
  } else {
 
    pred.binary.k <- as.integer(probs.k >= fold.cutoff.balanced.k)
 
    TP.k <- sum(pred.binary.k == 1 & true.k == 1)
    TN.k <- sum(pred.binary.k == 0 & true.k == 0)
    FP.k <- sum(pred.binary.k == 1 & true.k == 0)
    FN.k <- sum(pred.binary.k == 0 & true.k == 1)
 
    acc.k <- as.numeric((TP.k + TN.k) / length(true.k))
 
    mcc.denom.k <- sqrt((TP.k + FP.k) * (TP.k + FN.k) *
                        (TN.k + FP.k) * (TN.k + FN.k))
    mcc.k <- as.numeric(ifelse(mcc.denom.k == 0, 0,
                               (TP.k * TN.k - FP.k * FN.k) / mcc.denom.k))
 
    f1.k  <- as.numeric(ifelse((2*TP.k + FP.k + FN.k) == 0, 0,
                               (2*TP.k) / (2*TP.k + FP.k + FN.k)))
 
    cat(sprintf("Outer fold %d — AUC: %.4f  ACC: %.4f  MCC: %.4f  F1: %.4f  cutoff(balanced): %.6f\n",
                k, auc.k, acc.k, mcc.k, f1.k, fold.cutoff.balanced.k))
  }
 
  df.fold.metrics <- rbind(df.fold.metrics,
                           data.frame(fold            = k,
                                      auc             = as.numeric(auc.k),
                                      acc             = as.numeric(acc.k),
                                      mcc             = as.numeric(mcc.k),
                                      f1              = as.numeric(f1.k),
                                      cutoff.tnr90    = fold.cutoff.k,
                                      cutoff.balanced = fold.cutoff.balanced.k,
                                      ntrees          = var.ntrees.k,
                                      max_depth       = var.max_depth.k,
                                      learn_rate      = var.learn_rate.k,
                                      col_sample_rate = var.col_sample_rate.k))
 
  h2o.rm(hex.outer.train)
  h2o.rm(hex.outer.test)
  h2o.rm(gbm.model.k)
 
  for (mid in grid.k@model_ids) {
    h2o.rm(mid)
  }
  h2o.rm(grid.id.k)
 
}  
 

# Aggregated performance across all outer folds
cat("NESTED CV AGGREGATED PERFORMANCE\n")

pred.agg <- prediction(all.probs, all.true)
auc.agg  <- performance(pred.agg, "auc")@y.values[[1]]
cat(sprintf("Aggregated AUC (pooled): %.4f\n", auc.agg))
 

valid.folds <- !is.na(df.fold.metrics$cutoff.balanced)
n.valid     <- sum(valid.folds)
n.total     <- nrow(df.fold.metrics)
cat(sprintf("All metrics averaged over %d/%d folds with valid cutoff.balanced\n",
            n.valid, n.total))
 
cat(sprintf("Mean per-fold AUC: %.4f +/- %.4f\n",
            mean(df.fold.metrics$auc[valid.folds]), sd(df.fold.metrics$auc[valid.folds])))
cat(sprintf("Mean per-fold ACC: %.4f +/- %.4f\n",
            mean(df.fold.metrics$acc[valid.folds]), sd(df.fold.metrics$acc[valid.folds])))
cat(sprintf("Mean per-fold MCC: %.4f +/- %.4f\n",
            mean(df.fold.metrics$mcc[valid.folds]), sd(df.fold.metrics$mcc[valid.folds])))
cat(sprintf("Mean per-fold F1:  %.4f +/- %.4f\n",
            mean(df.fold.metrics$f1[valid.folds]),  sd(df.fold.metrics$f1[valid.folds])))
 
write.csv(df.fold.metrics,
          file = "nested_cv_fold_metrics.csv",
          row.names = FALSE, quote = FALSE)
 
df.nested.preds <- data.frame(true        = all.true,
                               probability = all.probs)
write.csv(df.nested.preds,
          file = "nested_cv_all_predictions.csv",
          row.names = FALSE, quote = FALSE)

# TNR vs TPR plot on aggregated predictions 
perf.agg <- performance(pred.agg, "tpr", "tnr")
 
pdf("TNRvsTPR_nestedCV.pdf", width = 8, height = 8)
par(mar = c(5, 5, 4, 2))
plot(perf.agg,
     avg              = "threshold",
     colorize         = TRUE,
     lwd              = 3,
     print.cutoffs.at = seq(0, 1, by = 0.05),
     text.adj         = c(-0.5, 0.5),
     text.cex         = 0.6)
grid(col = "lightgray")
axis(1, at = seq(0, 1, by = 0.1))
axis(2, at = seq(0, 1, by = 0.1))
abline(v = c(0.1, 0.3, 0.5, 0.7, 0.9), col = "lightgray", lty = "dotted")
abline(h = c(0.1, 0.3, 0.5, 0.7, 0.9), col = "lightgray", lty = "dotted")
lines(x = c(0, 1), y = c(0, 1), col = "black", lty = "dotted")
dev.off()
 
# FINAL MODEL: retrain on ALL data
final.ntrees          <- as.numeric(names(sort(table(df.fold.metrics$ntrees),
                                               decreasing = TRUE)[1]))
final.max_depth       <- as.numeric(names(sort(table(df.fold.metrics$max_depth),
                                               decreasing = TRUE)[1]))
final.learn_rate      <- mean(df.fold.metrics$learn_rate)
final.col_sample_rate <- mean(df.fold.metrics$col_sample_rate)
 
cat(sprintf("\nFinal model params (consensus): ntrees=%d, depth=%d, lr=%.4f, col_sample=%.4f\n",
            final.ntrees, final.max_depth, final.learn_rate, final.col_sample_rate))
 
write(c(final.ntrees, final.max_depth, final.learn_rate, final.col_sample_rate),
      file = "best_parameters.txt", ncolumns = 1)
 
gbm.model.final <- h2o.gbm(
  x                      = train.names,
  y                      = "response",
  training_frame         = data.hex,
  model_id               = model.id,
  balance_classes        = TRUE,
  max_after_balance_size = 5,
  stopping_metric        = "AUC",
  stopping_rounds        = 3,
  stopping_tolerance     = 0.001,
  ntrees                 = final.ntrees,
  max_depth              = final.max_depth,
  learn_rate             = final.learn_rate,
  col_sample_rate        = final.col_sample_rate
)
 
h2o.saveModel(object = gbm.model.final, path = path, force = TRUE)
print(path)
 
# Feature importance
df.fi <- as.data.frame(h2o.varimp(gbm.model.final))
write.csv(data.frame(feature    = df.fi$variable,
                     importance = df.fi$percentage),
          file = paste0(model.id, "_featureImportance.csv"),
          row.names = FALSE, quote = FALSE)
 

# Cutoff selection on 75/25 split 
set.seed(42)
n.train     <- round(0.75 * nrow(data.hex))
vec.indices <- 1:nrow(data.hex)
vec.train   <- sort(sample(vec.indices, n.train))
vec.test    <- vec.indices[!(vec.indices %in% vec.train)]
 
df.full        <- as.data.frame(data.hex)
data.hex.train <- as.h2o(df.full[vec.train, ])
data.hex.test  <- as.h2o(df.full[vec.test,  ])
data.hex.train$response <- as.factor(data.hex.train$response)
data.hex.test$response  <- as.factor(data.hex.test$response)
 
h2o.exportFile(data.hex.train,
               path = paste(path, "retrain_hex_train.csv", sep = "/"))
h2o.exportFile(data.hex.test,
               path = paste(path, "retrain_hex_test.csv",  sep = "/"))
 
gbm.model.final.75 <- h2o.gbm(
  x                      = train.names,
  y                      = "response",
  training_frame         = data.hex.train,
  model_id               = model.cutoff.id,
  balance_classes        = TRUE,
  max_after_balance_size = 5,
  stopping_metric        = "AUC",
  stopping_rounds        = 3,
  stopping_tolerance     = 0.001,
  ntrees                 = final.ntrees,
  max_depth              = final.max_depth,
  learn_rate             = final.learn_rate,
  col_sample_rate        = final.col_sample_rate
)
h2o.saveModel(object = gbm.model.final.75, path = path, force = TRUE)
print(path)
 
y.test.vals <- as.numeric(as.character(df.full[vec.test, "response"]))
preds.75    <- h2o.predict(gbm.model.final.75, data.hex.test)
probs.75    <- as.numeric(as.vector(preds.75$p1))
 
df.predictions.true <- data.frame(probabilities = probs.75,
                                   true          = y.test.vals)
 
pred.cutoff  <- prediction(df.predictions.true$probabilities,
                            df.predictions.true$true)
perf.cutoff  <- performance(pred.cutoff, "tpr", "tnr")
 
vec.tnr    <- unlist(perf.cutoff@x.values)
vec.tpr    <- unlist(perf.cutoff@y.values)
vec.cutoff <- unlist(perf.cutoff@alpha.values)
 
# Cutoff 1: TNR >= 0.9
cutoff.tnr       <- 0.9
vec.dist         <- abs(vec.tnr - cutoff.tnr)
vec.ind.cutoff   <- tail(which(vec.dist == min(vec.dist)), 1)
final.cutoff     <- vec.cutoff[vec.ind.cutoff]
 
# Cutoff 2: balanced
mat.bisectrix        <- cbind(seq(0, 1, 0.001), seq(0, 1, 0.001))
mat.points           <- cbind(vec.tnr, vec.tpr)
mat.dist.bisectrix   <- dist2Line(mat.points, mat.bisectrix)
mat.dist.bis.sorted  <- mat.dist.bisectrix[order(mat.dist.bisectrix[, 1]), ]
 
df.tnr.tpr <- data.frame(tnr    = vec.tnr,
                          tpr    = vec.tpr,
                          cutoff = vec.cutoff)
 
vec.dist.tnr <- abs(df.tnr.tpr$tnr - as.numeric(mat.dist.bis.sorted[1, 2]))
vec.dist.tpr <- abs(df.tnr.tpr$tpr - as.numeric(mat.dist.bis.sorted[1, 3]))
 
final.cutoff.balanced <- df.tnr.tpr[
  df.tnr.tpr$tnr == unique(df.tnr.tpr$tnr[which(vec.dist.tnr == min(vec.dist.tnr))]) &
  df.tnr.tpr$tpr == unique(df.tnr.tpr$tpr[which(vec.dist.tpr == min(vec.dist.tpr))]),
  "cutoff"]
 
write(final.cutoff,
      file = paste0(model.cutoff.id, "_cutoff.txt"))
write(final.cutoff.balanced,
      file = paste0(model.cutoff.id, "_cutoff_balanced.txt"))
 

# TNR vs TPR plot — 75/25 model
pdf("TNRvsTPR.pdf", width = 8, height = 8)
par(mar = c(5, 5, 4, 2))
plot(perf.cutoff,
     avg              = "threshold",
     colorize         = TRUE,
     lwd              = 3,
     print.cutoffs.at = seq(0, 1, by = 0.05),
     text.adj         = c(-0.5, 0.5),
     text.cex         = 0.6)
grid(col = "lightgray")
axis(1, at = seq(0, 1, by = 0.1))
axis(2, at = seq(0, 1, by = 0.1))
abline(v = c(0.1, 0.3, 0.5, 0.7, 0.9), col = "lightgray", lty = "dotted")
abline(h = c(0.1, 0.3, 0.5, 0.7, 0.9), col = "lightgray", lty = "dotted")
lines(x = c(0, 1), y = c(0, 1), col = "black", lty = "dotted")
dev.off()
 

# Shutdown
h2o.shutdown(prompt = FALSE)
 
end.main <- proc.time()
cat(sprintf("\nScript duration: %.2f min\n", (end.main - start.main)[3] / 60))