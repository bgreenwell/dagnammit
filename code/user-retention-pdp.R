library(ranger)
library(pdp)
library(ggplot2)
library(patchwork)
library(xgboost)
library(vip)

theme_set(theme_bw())

# Function to fit an XGBoost model to the simulated retention data
fit_xgb <- function(trn, val, adj = NULL) {
  if (!is.null(adj)) {
    trn <- trn[, adj]
    val <- val[, adj]    
  }
  X.trn <- data.matrix(subset(trn, select = -Did.renew))
  X.val <- data.matrix(subset(val, select = -Did.renew))
  y.trn <- trn$Did.renew
  y.val <- val$Did.renew
  dtrn <- xgb.DMatrix(X.trn, label = y.trn)
  dval <- xgb.DMatrix(X.val, label = y.val)
  watchlist <- list(trn = dtrn, val = dval)
  param <- list(max_depth = 2, eta = 0.001, verbose = 0, 
                objective = "binary:logistic", eval_metric = "logloss")
  xgb.train(param, data = dtrn, nrounds = 20000, watchlist = watchlist,
            early_stopping_rounds = 20)
}

# Load the simulated retention data
omit <- c("Product.need", "Bugs.faced")
# ret <- read.csv("/Users/b780620/Desktop/dss2024/retention.csv")[, keep]
ret <- read.csv("/Users/b780620/Desktop/dss2024/retention.csv")

# Set up training and validation sets
set.seed(1011)
trn.ids <- sample(nrow(ret), size = 8000, replace = FALSE)
ret.trn <- ret[trn.ids, ]
ret.val <- ret[-trn.ids, ]

# Remove variables we wouldn't realistically have
# ret.trn$Product.need <- NULL
# ret.trn$Bugs.faced <- NULL
# ret.val$Product.need <- NULL
# ret.val$Bugs.faced <- NULL


# Model fit ------


set.seed(1020)
bst <- fit_xgb(ret.trn, val = ret.val)
bst2 <- fit_xgb(ret.trn[, adjust.ad], val = ret.val[, adjust.ad])
# saveRDS(bst, file = "/Users/b780620/Desktop/devel/dagnammit/data/bst.rds")


# SHAP values --------

X.trn <- data.matrix(subset(ret.trn, select = -Did.renew))
shap <- predict(bst, newdata = X.trn, predcontrib = TRUE)[, -9L]

# Save SHAP output
# saveRDS(shap, file = "/Users/b780620/Desktop/devel/dagnammit/data/shap.rds")

# Use vip to plot SHAP-based VI scores
vi <- data.frame(
  "Variable" = colnames(shap),
  "Importance" = apply(shap, MARGIN = 2, FUN = function(x) mean(abs(x)))
)
vi <- tibble::as_tibble(vi)
class(vi) <- c("vi", class(vi))
vip::vip(vi, geom = "point")

barplot(apply(shap, MARGIN = 2, FUN = function(x) mean(abs(x))), horiz = TRUE)

# SHAP dependence plots
par(mfrow = c(2, 4), las = 1)
for (col in colnames(X.trn)) {
  color <- adjustcolor(1, alpha.f = 0.05)
  plot(X.trn[, col], y = shap[, col], ylim = range(shap), xlab = col,
       ylab = "SHAP value", col = color)
  quants <- quantile(X.trn[, col], probs = 1:9/10)
  rug(quants)
}


# PD plots ----------

# Prediction wrapper
pfun.xgb <- function(object, newdata) {
  mean(predict(object, newdata = newdata, outputmargin = FALSE))
}

# Use a subsample
set.seed(1043)
ids <- sample(nrow(ret.trn), size = 1000, replace = FALSE)
X.trn.samp <- data.matrix(subset(ret.trn[ids, ], select = -Did.renew))

# Loop through each feature, compute PD, and store results in a list
features <- names(subset(ret.trn, select = -Did.renew))
pds <- lapply(features, FUN = function(feature) {
  partial(
    bst, 
    pred.var = feature,
    # train = X.trn.samp,
    train = X.trn,
    pred.fun = pfun.xgb,
    quantiles = TRUE,
    probs = 1:29/30,
    progress = TRUE
  )
})

# Save PD output
# saveRDS(pds, file = "/Users/b780620/Desktop/devel/dagnammit/data/pds.rds")

# Display PD plots
pdps <- lapply(pds, FUN = function(p) {
  autoplot(p) + ylim(0, 1) + ylab("Partial dependence")
})
gridExtra::grid.arrange(grobs = pdps, nrow = 2)


########
pfun.truth <- function(object, newdata) {
  eta <-  
    1.26 * newdata[, "Product.need"] + 
    0.56 * newdata[, "Monthly.usage"] + 
    0.7 * newdata[, "Economy"] + 
    0.35 * newdata[, "Discount"] + 
    0.35 * (1 - newdata[, "Bugs.faced"] / 20) + 
    0.035 * newdata[, "Sales.calls"] + 
    0.105 * newdata[, "Interactions"] + 
    0.7 / (newdata[, "Last.upgrade"] / 4 + 0.25) - 3.15 
  mean(plogis(eta))
  # mean(eta)
  # plogis(mean(eta))
}
pds.truth <- lapply(features, FUN = function(feature) {
  partial(
    bst, 
    pred.var = feature,
    # train = X.trn.samp,
    train = X.trn,
    pred.fun = pfun.truth,
    quantiles = TRUE,
    probs = 1:29/30,
    progress = TRUE
  )
})
# saveRDS(pds.truth, file = "/Users/b780620/Desktop/devel/dagnammit/data/pds_truth.rds")
pdps.truth <- lapply(pds.truth, FUN = function(p) {
  autoplot(p) + ylim(0, 1) + ylab("Partial dependence")
})
gridExtra::grid.arrange(grobs = pdps.truth, nrow = 2)

pds <- readRDS("/Users/b780620/Desktop/devel/dagnammit/data/pds.rds")
names(pds) <- names(pds.truth) <- features
pdps.all <- lapply(features, FUN = function(feature) {
  autoplot(pds[[feature]]) + 
    geom_line(data = pds.truth[[feature]], color = "red") +
    ylim(0, 1) + 
    ylab("Partial dependence")
})
gridExtra::grid.arrange(grobs = pdps.all, nrow = 2)


########


# Adjusted PD plot for ad spend ------------------------------------------------

# Use an RF with appropriate adjustment set to estimate ATE of ad spend
adjust.ad <- c("Did.renew", "Ad.spend", "Monthly.usage", "Last.upgrade")
set.seed(1320)
bst.ad <- fit_xgb(ret.trn[, adjust.ad], val = ret.val[, adjust.ad])

set.seed(1105)
samp <- ret.trn[sample(nrow(ret.trn), size = 1000), 
                c("Ad.spend", "Monthly.usage", "Last.upgrade")]
palette("Okabe-Ito")
pairs(samp, col = adjustcolor(ret.trn$Did.renew + 1, alpha.f = 0.4))
palette("default")
saveRDS(samp, file = "/Users/b780620/Desktop/devel/dagnammit/data/samp.rds")


# rfo.ad <- ranger(Did.renew ~ ., data = ret.trn[, adjust.ad], probability = TRUE)
# pfun <- function(object, newdata) {
#   mean(predict(object, data = newdata)$predictions[, 2L])
#   # predict(object, data = newdata)$predictions[, 2L]
#   #quantile(qlogis(p), probs = 1:9/10)
# }
pd.ad <- partial(
  bst.ad, 
  pred.var = "Ad.spend", 
  train = X.trn,#[, adjust.ad],
  pred.fun = pfun.xgb, 
  #quantiles = TRUE,
  #probs = 1:29/30,
  progress = TRUE
)


# Save PD output
saveRDS(pd.ad, file = "/Users/b780620/Desktop/devel/dagnammit/data/pd_ad_adjusted.rds")

# PD plot for spend with proper adjustment set
#pdps[[7L]] + ggtitle("XGBoost") + 
  autoplot(pd.ad) + ylim(0, 1) + 
  ylab("Partial dependence") + ggtitle("Adjusted RFO")




# Double machine learning ------------------------------------------------------

# Specify the data and variables for the causal model
dml_data <- DoubleML::DoubleMLData$new(
  data = ret.trn,
  y_col = "Did.renew",
  d_cols = "Ad.spend",
  x_cols = c("Last.upgrade", "Monthly.usage")
)
print(dml_data)

lrnr <- mlr3::lrn("regr.ranger", num.trees = 500)
set.seed(1810)  # for reproducibility
dml_plr = DoubleML::DoubleMLPLR$new(
  dml_data, ml_l = lrnr$clone(), ml_m = lrnr$clone()
)
dml_plr$fit()

# Print results
print(dml_plr)
print(dml_plr$confint())


# ------------------ Fit summary       ------------------
#   Estimates and significance testing of the effect of target variables
#          Estimate. Std. Error t value Pr(>|t|)
# Ad.spend  -0.09634    0.25197  -0.382    0.702

# Confidence interval
print(dml_plr$confint())
#               2.5 %   97.5 %
# Ad.spend -0.5901917 0.397511

# Hack away at it manually
set.seed(1223)
fit.x <- ranger(Ad.spend ~ Monthly.usage + Last.upgrade, data = ret.trn)
fit.y <- ranger(Did.renew ~ Monthly.usage + Last.upgrade, data = ret.trn, probability = TRUE)
res.x <- ret.trn$Ad.spend - predict(fit.x, data = ret.trn)$predictions
res.y <- ret.trn$Did.renew - predict(fit.y, data = ret.trn)$predictions[, 2L]
fit.res <- ranger(res.y ~ res.x, data = data.frame(cbind(res.x, res.y)))
preds.res <- predict(fit.res, data = data.frame(cbind(res.x, res.y)))$predictions
summary(fit.lm <- lm(res.y ~ res.x))

palette("OkabeIto")
scatter.smooth(ret.trn$Ad.spend, y = preds.res, lpars = list(lwd = 3, col = 1),
               col = adjustcolor(2, alpha.f = 0.1), las = 1,
               xlab = "Ad spend", ylab = "Centered effect on renewal")
abline(h = 0, lwd = 2, col = 3, lty = 1)
legend("topright", inset = 0.01, legend = c("Truth", "Double ML"), 
       lty = c(1, 1), col = c(3, 1), lwd = 2, bty = "n")
# abline(fit.lm, col = 4, lwd = 2, lty = 3)
palette("default")
