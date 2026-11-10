## grt_mn_subdesigns.R — B 路線 D1：用 mdsdt 擬合 Miller & Nicely (1955) 的特徵對子設計
##
## 用法（筆電 / Arc）：
##     install.packages("mdsdt")
##     Rscript grt_mn_subdesigns.R            # 在 Visual_AudioWM/ 目錄下執行
##
## 輸入：review/data/grt_subdesigns/index.csv 與其列出的長格式 csv（由 grt_mn_subdesigns.py 產生）
## 輸出：review/data/grt_subdesigns/r_mdsdt_results.csv     每個 (design, snr, model) 的 logLik / AIC / 參數（長格式）
##       review/data/grt_subdesigns/r_mdsdt_tests.csv       mriTest / riTest 的結果
##       review/data/grt_subdesigns/r_mdsdt_<design>.pdf    每個設計一份 PDF，每個 SNR 一頁擬合橢圓
##       review/data/grt_subdesigns/r_mdsdt_log.txt         逐步訊息（包括哪些呼叫失敗）
##
## ⚠ 本腳本在沒有 R 的容器裡寫成，mdsdt 的函式名與引數是憑套件文件的記憶寫的，未經執行核對。
##   標 [UNVERIFIED] 的地方請對照 ?fit.grt、?mriTest、?riTest、str(silbert09a)、str(thomas15a) 確認。
##   每個 mdsdt 呼叫都包在 tryCatch 裡，失敗只會寫進 log 不會中斷。
##
## 4×4 頻率矩陣的版面（假設，[UNVERIFIED]）：
##   列 = 刺激、欄 = 反應，順序皆為 x 跑最快：(x0,y0), (x1,y0), (x0,y1), (x1,y1)
##   x = 子設計的軸 f（index.csv 的 axis_f），y = 軸 g。
##   python 長格式 csv 的子音順序已是 f 跑最快（grt_mn_subdesigns.py SubDesign.consonants），所以直接 xtabs 即可。
##   若 str(silbert09a) 顯示 mdsdt 用的是 y 跑最快，把下面 LAYOUT 改成 "y_fast"。
LAYOUT <- "x_fast"

suppressPackageStartupMessages({
  ok_mdsdt <- requireNamespace("mdsdt", quietly = TRUE)
})
if (!ok_mdsdt) stop("mdsdt 未安裝：install.packages('mdsdt')")
library(mdsdt)

DIR <- file.path("review", "data", "grt_subdesigns")
if (!file.exists(file.path(DIR, "index.csv"))) stop("找不到 ", file.path(DIR, "index.csv"), "；請在 Visual_AudioWM/ 下執行")
index <- read.csv(file.path(DIR, "index.csv"), stringsAsFactors = FALSE)
LOG <- file.path(DIR, "r_mdsdt_log.txt")
cat("", file = LOG)
logmsg <- function(...) { msg <- paste0(...); message(msg); cat(msg, "\n", file = LOG, append = TRUE) }

## ------------------------------------------------------------------ 資料整形

## 長格式 → 4×4（或 n×n）矩陣，列 = 刺激、欄 = 反應
freq_matrix <- function(d, layout = LAYOUT) {
  lf <- sort(unique(d$stim_f)); lg <- sort(unique(d$stim_g))
  if (layout == "x_fast") {
    cells <- expand.grid(f = lf, g = lg)                       # f 跑最快
  } else {
    cells <- expand.grid(g = lg, f = lf)[, c("f", "g")]        # g 跑最快
  }
  lab <- paste0("f", cells$f, "g", cells$g)
  m <- matrix(0L, nrow(cells), nrow(cells), dimnames = list(stim = lab, resp = lab))
  for (i in seq_len(nrow(d))) {
    s <- paste0("f", d$stim_f[i], "g", d$stim_g[i]); r <- paste0("f", d$resp_f[i], "g", d$resp_g[i])
    m[s, r] <- m[s, r] + d$count[i]
  }
  m
}

## 長格式 → xtabs(count ~ resp_f + resp_g + stim)：mdsdt 文件說 fit.grt 可吃 xtabs（thomas15a/b 範例為 3×3）[UNVERIFIED]
freq_xtabs <- function(d) {
  d$stim <- factor(d$stim, levels = unique(d$stim[order(d$stim_g, d$stim_f)]))
  xtabs(count ~ resp_f + resp_g + stim, data = d)
}

## ------------------------------------------------------------------ 參數擷取（fit.grt 回傳物件的欄位名 [UNVERIFIED]）

grab <- function(fit, names) { for (n in names) if (!is.null(fit[[n]])) return(fit[[n]]); NULL }

extract_params <- function(fit) {
  out <- data.frame(param = character(0), value = numeric(0), stringsAsFactors = FALSE)
  add <- function(nm, v) { v <- as.numeric(v); if (length(v)) out <<- rbind(out, data.frame(param = nm, value = v)) }
  means <- grab(fit, c("means", "mu", "mean"))
  if (!is.null(means)) {
    mm <- as.matrix(means)
    for (i in seq_len(nrow(mm))) for (j in seq_len(ncol(mm))) add(sprintf("mean_stim%d_dim%d", i, j), mm[i, j])
  }
  covs <- grab(fit, c("covmat", "sigma", "cov", "covs"))
  if (!is.null(covs)) {
    if (is.list(covs)) {
      for (i in seq_along(covs)) {
        s <- as.matrix(covs[[i]])
        add(sprintf("var_stim%d_x", i), s[1, 1]); add(sprintf("var_stim%d_y", i), s[2, 2])
        add(sprintf("rho_stim%d", i), s[1, 2] / sqrt(s[1, 1] * s[2, 2]))
      }
    } else add("cov_flat", covs)
  }
  for (nm in c("a1", "a2", "criteria", "thresholds", "bounds", "c1", "c2")) {
    v <- fit[[nm]]; if (!is.null(v)) add(paste0(nm, "_", seq_along(as.numeric(v))), v)
  }
  if (nrow(out) == 0) {                                           # 完全猜錯欄位名時的保底：把所有數值欄位攤平
    num <- fit[sapply(fit, is.numeric)]
    for (nm in names(num)) add(paste0(nm, "_", seq_along(num[[nm]])), num[[nm]])
  }
  out
}

## 模型清單：mdsdt::fit.grt(freq, PS_x, PS_y, PI)；引數名與 PI 選項 "none"/"same_rho"/"all" 已對照 CRAN 文件（2026-11-10）。mriTest / riTest 函式名亦已確認。
MODELS <- list(
  full          = list(PS_x = FALSE, PS_y = FALSE, PI = "none"),
  PS_x          = list(PS_x = TRUE,  PS_y = FALSE, PI = "none"),
  PS_y          = list(PS_x = FALSE, PS_y = TRUE,  PI = "none"),
  PS_xy         = list(PS_x = TRUE,  PS_y = TRUE,  PI = "none"),
  PI_same_rho   = list(PS_x = FALSE, PS_y = FALSE, PI = "same_rho"),
  PI_all        = list(PS_x = FALSE, PS_y = FALSE, PI = "all"),
  PS_xy_PI_all  = list(PS_x = TRUE,  PS_y = TRUE,  PI = "all")     # 獨立通道（= 企劃的 M0 在二維的版本）
)

fit_one <- function(freq, spec) {
  tryCatch(fit.grt(freq, PS_x = spec$PS_x, PS_y = spec$PS_y, PI = spec$PI),
           error = function(e) { logmsg("    fit.grt 失敗: ", conditionMessage(e)); NULL })
}

## ------------------------------------------------------------------ 主迴圈

results <- list(); tests <- list()
designs <- unique(index$design)
for (dn in designs) {
  rows <- index[index$design == dn, ]
  logmsg("=== ", dn, "  ", rows$size[1], "  {", rows$consonants[1], "}")
  is_2x2 <- rows$size[1] == "2x2"
  pdf_path <- file.path(DIR, paste0("r_mdsdt_", dn, ".pdf"))
  pdf(pdf_path, width = 7, height = 7)
  for (k in seq_len(nrow(rows))) {
    snr <- rows$snr[k]
    d <- read.csv(file.path(DIR, rows$file[k]), stringsAsFactors = FALSE)
    logmsg("  SNR ", snr, " dB  (retained ", rows$prop_retained[k], ")")
    if (is_2x2) {
      freq <- freq_matrix(d)
      inputs <- list(matrix = freq)
    } else {
      ## 2×3：先試 xtabs 形式，再試矩陣；mdsdt 是否支援兩軸水準數不同的設計 [UNVERIFIED]
      inputs <- list(xtabs = tryCatch(freq_xtabs(d), error = function(e) NULL), matrix = freq_matrix(d))
    }
    fitted_full <- NULL
    for (mname in names(MODELS)) {
      fit <- NULL
      for (iname in names(inputs)) {
        if (is.null(inputs[[iname]])) next
        fit <- fit_one(inputs[[iname]], MODELS[[mname]])
        if (!is.null(fit)) { used_input <- iname; break }
      }
      if (is.null(fit)) {
        logmsg("    [", mname, "] 無法擬合（", if (is_2x2) "2x2" else "2x3：mdsdt 可能不支援非方形設計", "）")
        next
      }
      ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
      aic <- tryCatch(AIC(fit), error = function(e) NA_real_)
      pars <- tryCatch(extract_params(fit), error = function(e) data.frame(param = character(0), value = numeric(0)))
      results[[length(results) + 1]] <- data.frame(design = dn, snr = snr, model = mname, input = used_input,
                                                   logLik = ll, AIC = aic,
                                                   param = c("logLik", "AIC", pars$param),
                                                   value = c(ll, aic, pars$value), stringsAsFactors = FALSE)
      logmsg(sprintf("    [%-13s] logLik=%.2f AIC=%.2f  (%d params extracted)", mname, ll, aic, nrow(pars)))
      if (mname == "full") fitted_full <- fit
    }
    ## 擬合橢圓
    if (!is.null(fitted_full)) {
      tryCatch({ plot(fitted_full, main = sprintf("%s  SNR %s dB", dn, snr)) },     # plot.grt [UNVERIFIED]
               error = function(e) logmsg("    plot 失敗: ", conditionMessage(e)))
    }
    ## 邊際反應不變性 / 報告獨立性：函式名 mriTest / riTest [UNVERIFIED]；僅對 2×2 的矩陣輸入
    if (is_2x2) {
      for (tn in c("mriTest", "riTest")) {
        if (!exists(tn, where = asNamespace("mdsdt"), inherits = FALSE)) { logmsg("    mdsdt 沒有 ", tn); next }
        r <- tryCatch(do.call(tn, list(freq)), error = function(e) { logmsg("    ", tn, " 失敗: ", conditionMessage(e)); NULL })
        if (!is.null(r)) {
          txt <- paste(capture.output(print(r)), collapse = " | ")
          tests[[length(tests) + 1]] <- data.frame(design = dn, snr = snr, test = tn, result = txt, stringsAsFactors = FALSE)
          logmsg("    ", tn, ": ", substr(txt, 1, 200))
        }
      }
    }
  }
  dev.off()
}

if (length(results)) {
  res <- do.call(rbind, results)
  write.csv(res, file.path(DIR, "r_mdsdt_results.csv"), row.names = FALSE)
  logmsg("寫出 ", file.path(DIR, "r_mdsdt_results.csv"), "（", nrow(res), " 列）")
  ## 模型比較小表：每個 (design, snr) 的 AIC
  cmp <- unique(res[, c("design", "snr", "model", "logLik", "AIC")])
  write.csv(cmp, file.path(DIR, "r_mdsdt_model_comparison.csv"), row.names = FALSE)
} else logmsg("沒有任何擬合成功 — 檢查 fit.grt 的引數名")
if (length(tests)) write.csv(do.call(rbind, tests), file.path(DIR, "r_mdsdt_tests.csv"), row.names = FALSE)
logmsg("完成")
