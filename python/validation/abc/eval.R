library(arrow)
library(transport)

base_output <- "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/validation/abc"
result_dir <- file.path(base_output, "abc_results")

populations <- c("CEU", "CHB", "YRI")
sizes <- c(5, 10, 15, 20)

full_medians <- list()
for (pop in populations) {
  post_path <- file.path(result_dir, paste0("posteriors_", pop, ".csv"))
  p <- read.csv(post_path)
  p <- p[p$type == "full", ]
  full_medians[[pop]] <- list(T = median(p$T), N0 = median(p$N0))
}

wd_files <- file.path(result_dir, paste0("wd_raw_", populations, ".csv"))
wd_raw <- do.call(rbind, lapply(wd_files, read.csv))

sub_df <- wd_raw[wd_raw$type == "subsample", ]
tf_df <- wd_raw[wd_raw$type == "transform", ]

sub_agg <- aggregate(cbind(wd_sub_T = wd_T, wd_sub_N0 = wd_N0) ~ target + n, data = sub_df, FUN = mean)
tf_agg <- aggregate(cbind(wd_tf_T = wd_T, wd_tf_N0 = wd_N0) ~ target + reference + n, data = tf_df, FUN = mean)

merged <- merge(tf_agg, sub_agg, by = c("target", "n"))
merged$ratio_T <- merged$wd_sub_T / merged$wd_tf_T
merged$ratio_N0 <- merged$wd_sub_N0 / merged$wd_tf_N0
merged <- merged[order(merged$target, merged$reference, merged$n), ]

cat("WD: transform vs subsample to full \n\n")
cat("target,ref,n,wd_sub_T,wd_tf_T,ratio_T,wd_sub_N0,wd_tf_N0,ratio_N0\n")
for (i in seq_len(nrow(merged))) {
  r <- merged[i, ]
  cat(sprintf("%s,%s,%d,%.1f,%.1f,%.3f,%.1f,%.1f,%.3f\n",
              r$target, r$reference, r$n,
              r$wd_sub_T, r$wd_tf_T, r$ratio_T,
              r$wd_sub_N0, r$wd_tf_N0, r$ratio_N0))
}

cat("\n\nBias relative to full-data posterior median \n\n")

post_files <- file.path(result_dir, paste0("posteriors_", populations, ".csv"))
post_raw <- do.call(rbind, lapply(post_files, read.csv))

post_sub_tf <- post_raw[post_raw$type %in% c("subsample", "transform"), ]

post_med <- aggregate(cbind(median_T = T, median_N0 = N0) ~ target + type + reference + n,
                      data = post_sub_tf, FUN = median)

full_comb <- data.frame(
  target = names(full_medians),
  full_median_T = sapply(full_medians, function(x) x$T),
  full_median_N0 = sapply(full_medians, function(x) x$N0)
)

tbl <- merge(post_med, full_comb, by = "target")
tbl$bias_T_vs_full <- (tbl$median_T - tbl$full_median_T) / tbl$full_median_T
tbl$bias_N0_vs_full <- (tbl$median_N0 - tbl$full_median_N0) / tbl$full_median_N0
tbl <- tbl[order(tbl$target, tbl$reference, tbl$type, tbl$n), ]

cat("target,ref,method,n,bias_T_vs_full,bias_N0_vs_full\n")
for (i in seq_len(nrow(tbl))) {
  r <- tbl[i, ]
  cat(sprintf("%s,%s,%s,%d,%+.4f,%+.4f\n",
              r$target, r$reference, r$type, r$n,
              r$bias_T_vs_full, r$bias_N0_vs_full))
}

write.csv(tbl, file.path(result_dir, "abc_summary_table.csv"), row.names = FALSE)

summ_rows <- list()
for (pop in populations) {
  p <- post_raw[post_raw$target == pop, ]
  for (param in c("T", "N0")) {
    param_label <- if (param == "T") "Divergence time (T)" else "Population size (N0)"
    types <- unique(p$type)
    for (ty in types) {
      sub <- p[p$type == ty, ]
      if (ty == "full") {
        vals <- sub[[param]]
        summ_rows[[length(summ_rows) + 1]] <- data.frame(
          target = pop, param = param_label,
          method = "Full data", n_lab = "Full",
          q025 = quantile(vals, 0.025),
          q25 = quantile(vals, 0.25),
          q50 = quantile(vals, 0.50),
          q75 = quantile(vals, 0.75),
          q975 = quantile(vals, 0.975)
        )
      } else if (ty == "subsample") {
        for (sz in sizes) {
          vals <- sub[sub$n == sz, param]
          if (length(vals) == 0) next
          summ_rows[[length(summ_rows) + 1]] <- data.frame(
            target = pop, param = param_label,
            method = "Subsample", n_lab = as.character(sz),
            q025 = quantile(vals, 0.025),
            q25 = quantile(vals, 0.25),
            q50 = quantile(vals, 0.50),
            q75 = quantile(vals, 0.75),
            q975 = quantile(vals, 0.975)
          )
        }
      } else if (ty == "transform") {
        refs <- unique(sub$reference)
        for (ref in refs) {
          ref_sub <- sub[sub$reference == ref, ]
          for (sz in sizes) {
            vals <- ref_sub[ref_sub$n == sz, param]
            if (length(vals) == 0) next
            summ_rows[[length(summ_rows) + 1]] <- data.frame(
              target = pop, param = param_label,
              method = paste0("Transform (", ref, ")"),
              n_lab = as.character(sz),
              q025 = quantile(vals, 0.025),
              q25 = quantile(vals, 0.25),
              q50 = quantile(vals, 0.50),
              q75 = quantile(vals, 0.75),
              q975 = quantile(vals, 0.975)
            )
          }
        }
      }
    }
  }
}

posterior_summary <- do.call(rbind, summ_rows)
rownames(posterior_summary) <- NULL
write.csv(posterior_summary, file.path(result_dir, "posterior_summary.csv"), row.names = FALSE)

cat("Wrote posterior_summary.csv\n")
cat("\nDone.\n")