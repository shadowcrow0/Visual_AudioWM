## grt_mn_subdesigns.R — B 路線 D1：用 mdsdt 擬合 Miller & Nicely (1955) 的特徵對子設計
##
## 用法（筆電 / Arc）：
##     install.packages("mdsdt")
##     Rscript grt_mn_subdesigns.R            # 在 Visual_AudioWM/ 目錄下執行
##
## 輸入：review/data/grt_subdesigns/index.csv 與其列出的長格式 csv（由 grt_mn_subdesigns.py 產生）
## 輸出：review/data/grt_subdesigns/r_mdsdt_results.csv           每個 (design, sub, snr, model) 的 AIC / BIC / 參數（長格式）
##       review/data/grt_subdesigns/r_mdsdt_model_comparison.csv  每個 (design, sub, snr) 的模型 AIC 小表
##       review/data/grt_subdesigns/r_mdsdt_tests.csv             mriTest / riTest 的 z、χ²、p
##       review/data/grt_subdesigns/r_mdsdt_<design>.pdf          每個設計一份 PDF，每個 SNR 一頁擬合橢圓
##       review/data/grt_subdesigns/r_mdsdt_log.txt               逐步訊息
##
## 2026-11-10 已對照 mdsdt 1.3 原始碼（R/grt_base.R，GitHub hawkrobe/mdsdt）：
##   * fit.grt(freq, PS_x=FALSE, PS_y=FALSE, PI="none"|"same_rho"|"all", method=NA)
##   * 4×4 輸入的列／欄順序必須是 a1b1, a1b2, a2b1, a2b2 —— 第二個維度 b 跑最快、a 慢。
##     a 對應 PS_x（x = 慢的維度）、b 對應 PS_y。這裡 a = 軸 f、b = 軸 g。
##   * 回傳 S3 "grt"：$dists（每刺激一列：mu_r, sd_r, mu_c, sd_c, rho）、$AIC、$AIC.c、$BIC、
##     $fit$loglik（注意 2×2 路徑存的是 −logLik，n×n 路徑存的是 logLik；比較請用 $AIC）、$fit$iter、$fit$code（只有 n×n）。
##   * mriTest(freq) / riTest(freq) 吃 4×4 矩陣，不是 fit 物件。
##   * create_n_by_n_mod 假設兩軸水準數相同（nspd = sqrt(n_stim)），所以 2×3 設計不能直接餵；
##     本腳本把 2×3 切成相鄰水準的兩個 2×2（再一次條件化在子集內的反應上）。
##   * 2×2 一律餵 4×4 矩陣；長度 16 的 xtabs 會被誤導到 2×2 路徑而出錯。
## 本容器的沙盒不允許安裝第三方 R 套件，所以腳本只做過語法檢查（parse），尚未實跑。

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

## 長格式（兩軸各兩個水準）→ 4×4 矩陣，列 = 刺激、欄 = 反應，順序 f0g0, f0g1, f1g0, f1g1（g 跑最快 = mdsdt 的 a1b1,a1b2,a2b1,a2b2）
freq_matrix <- function(d) {
  lf <- sort(unique(c(d$stim_f, d$resp_f))); lg <- sort(unique(c(d$stim_g, d$resp_g)))
  stopifnot(length(lf) == 2, length(lg) == 2)
  cells <- expand.grid(g = lg, f = lf)[, c("f", "g")]            # g 跑最快
  lab <- paste0("f", cells$f, "g", cells$g)
  m <- matrix(0, nrow(cells), nrow(cells), dimnames = list(stim = lab, resp = lab))
  for (i in seq_len(nrow(d))) {
    s <- paste0("f", d$stim_f[i], "g", d$stim_g[i]); r <- paste0("f", d$resp_f[i], "g", d$resp_g[i])
    m[s, r] <- m[s, r] + d$count[i]
  }
  m
}

## 2×3 → 相鄰水準的 2×2 子表（只留刺激與反應都落在那兩個水準的列）
split_2x3 <- function(d, levels_f, levels_g) {
  lf <- as.integer(strsplit(levels_f, " ")[[1]]); lg <- as.integer(strsplit(levels_g, " ")[[1]])
  if (length(lf) == 2 && length(lg) == 2) return(list(all = d))
  out <- list()
  if (length(lg) == 3) {
    for (k in 1:2) {
      keep <- lg[k:(k + 1)]
      sub <- d[d$stim_g %in% keep & d$resp_g %in% keep, ]
      out[[sprintf("g%d%d", keep[1], keep[2])]] <- sub
    }
  } else {
    for (k in 1:2) {
      keep <- lf[k:(k + 1)]
      sub <- d[d$stim_f %in% keep & d$resp_f %in% keep, ]
      out[[sprintf("f%d%d", keep[1], keep[2])]] <- sub
    }
  }
  out
}

## ------------------------------------------------------------------ 參數擷取

extract_params <- function(fit) {
  out <- data.frame(param = character(0), value = numeric(0), stringsAsFactors = FALSE)
  add <- function(nm, v) { v <- as.numeric(v); if (length(v) && !is.na(v)) out <<- rbind(out, data.frame(param = nm, value = v)) }
  dists <- fit$dists
  if (!is.null(dists)) {
    dists <- as.matrix(dists)
    stim_lab <- if (!is.null(rownames(dists))) rownames(dists) else paste0("stim", seq_len(nrow(dists)))
    for (i in seq_len(nrow(dists))) for (j in seq_len(ncol(dists)))
      add(sprintf("%s_%s", colnames(dists)[j], stim_lab[i]), dists[i, j])
  }
  add("rowcuts", fit$rowcuts); add("colcuts", fit$colcuts)
  add("AIC", fit$AIC); add("AIC.c", fit$AIC.c); add("BIC", fit$BIC)
  add("fit_loglik_raw", fit$fit$loglik)       # 2×2 路徑為 −logLik，n×n 為 logLik；原樣記下
  add("iter", if (is.list(fit$fit$iter)) NA else fit$fit$iter[1])
  add("code", fit$fit$code)
  out
}

## 模型清單（引數名與 PI 選項已對照原始碼）
MODELS <- list(
  full          = list(PS_x = FALSE, PS_y = FALSE, PI = "none"),
  PS_x          = list(PS_x = TRUE,  PS_y = FALSE, PI = "none"),      # 軸 f 知覺可分
  PS_y          = list(PS_x = FALSE, PS_y = TRUE,  PI = "none"),      # 軸 g 知覺可分
  PS_xy         = list(PS_x = TRUE,  PS_y = TRUE,  PI = "none"),
  PI_same_rho   = list(PS_x = FALSE, PS_y = FALSE, PI = "same_rho"),
  PI_all        = list(PS_x = FALSE, PS_y = FALSE, PI = "all"),       # 知覺獨立（ρ = 0）
  PS_xy_PI_all  = list(PS_x = TRUE,  PS_y = TRUE,  PI = "all")        # 獨立通道（= 企劃的 M0 在二維的版本）
)

fit_one <- function(freq, spec) {
  ## 警告（例如 PI_same_rho 的 "NaNs produced"）只記 log 不中斷；錯誤（共變異矩陣非正定等）記 log、回傳 NULL
  tryCatch(
    withCallingHandlers({
      f <- fit.grt(freq, PS_x = spec$PS_x, PS_y = spec$PS_y, PI = spec$PI)
      if (identical(f, FALSE)) stop("checkConfusionMatrix 回傳 FALSE")
      f
    }, warning = function(w) { logmsg("    fit.grt 警告: ", conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) { logmsg("    fit.grt 失敗: ", conditionMessage(e)); NULL })
}

## ------------------------------------------------------------------ 主迴圈

results <- list(); tests <- list()
designs <- unique(index$design)
for (dn in designs) {
  rows <- index[index$design == dn, ]
  logmsg("=== ", dn, "  ", rows$size[1], "  {", rows$consonants[1], "}  f=", rows$axis_f[1], " g=", rows$axis_g[1])
  pdf_path <- file.path(DIR, paste0("r_mdsdt_", dn, ".pdf"))
  pdf(pdf_path, width = 7, height = 7)
  for (k in seq_len(nrow(rows))) {
    snr <- rows$snr[k]
    d <- read.csv(file.path(DIR, rows$file[k]), stringsAsFactors = FALSE)
    subs <- split_2x3(d, rows$levels_f[k], rows$levels_g[k])
    for (sname in names(subs)) {
      dd <- subs[[sname]]
      freq <- freq_matrix(dd)
      logmsg("  SNR ", snr, " dB  sub=", sname, "  N=", sum(freq), "  (設計保留率 ", rows$prop_retained[k], ")")
      fitted_full <- NULL
      for (mname in names(MODELS)) {
        fit <- fit_one(freq, MODELS[[mname]])
        if (is.null(fit)) { logmsg("    [", mname, "] 無法擬合"); next }
        aic <- suppressWarnings(as.numeric(fit$AIC)); bic <- suppressWarnings(as.numeric(fit$BIC))
        pars <- tryCatch(extract_params(fit), error = function(e) data.frame(param = character(0), value = numeric(0)))
        results[[length(results) + 1]] <- data.frame(design = dn, sub = sname, snr = snr, model = mname,
                                                     AIC = aic, BIC = bic,
                                                     param = pars$param, value = pars$value, stringsAsFactors = FALSE)
        logmsg(sprintf("    [%-13s] AIC=%.2f BIC=%.2f", mname, aic, bic))
        if (mname == "full") fitted_full <- fit
      }
      if (!is.null(fitted_full)) {
        tryCatch(plot(fitted_full, main = sprintf("%s [%s]  SNR %s dB", dn, sname, snr),
                      xlab = rows$axis_f[1], ylab = rows$axis_g[1]),
                 error = function(e) logmsg("    plot 失敗: ", conditionMessage(e)))
      }
      ## 邊際反應不變性（z 檢定，Thomas 2001）與報告獨立性（χ²），皆吃 4×4 矩陣
      r <- tryCatch(mriTest(freq), error = function(e) { logmsg("    mriTest 失敗: ", conditionMessage(e)); NULL })
      if (is.data.frame(r)) for (i in seq_len(nrow(r)))
        tests[[length(tests) + 1]] <- data.frame(design = dn, sub = sname, snr = snr, test = "mriTest",
                                                 label = r$response[i], statistic = r$z[i], p = r$p.value[i], stringsAsFactors = FALSE)
      r <- tryCatch(riTest(freq), error = function(e) { logmsg("    riTest 失敗: ", conditionMessage(e)); NULL })
      if (is.data.frame(r)) for (i in seq_len(nrow(r)))
        tests[[length(tests) + 1]] <- data.frame(design = dn, sub = sname, snr = snr, test = "riTest",
                                                 label = r$stimulus[i], statistic = r$chi.2[i], p = r$p.value[i], stringsAsFactors = FALSE)
    }
  }
  dev.off()
}

if (length(results)) {
  res <- do.call(rbind, results)
  write.csv(res, file.path(DIR, "r_mdsdt_results.csv"), row.names = FALSE)
  logmsg("寫出 ", file.path(DIR, "r_mdsdt_results.csv"), "（", nrow(res), " 列）")
  cmp <- unique(res[, c("design", "sub", "snr", "model", "AIC", "BIC")])
  cmp <- cmp[order(cmp$design, cmp$sub, cmp$snr, cmp$AIC), ]
  write.csv(cmp, file.path(DIR, "r_mdsdt_model_comparison.csv"), row.names = FALSE)
} else logmsg("沒有任何擬合成功")
if (length(tests)) write.csv(do.call(rbind, tests), file.path(DIR, "r_mdsdt_tests.csv"), row.names = FALSE)
logmsg("完成")
