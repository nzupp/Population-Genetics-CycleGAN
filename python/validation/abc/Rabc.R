library(abc)
library(transport)
library(arrow)

# Split for speed again
args <- commandArgs(trailingOnly = TRUE)
target <- args[1]

ABC_TOL <- 0.01
csv_dir <- "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/validation/abc/abc_csvs"
base_output <- "/fs/ess/PAA0202/Zuppas/Dissertation/Chapter1/validation/abc"

populations <- c("CEU", "YRI", "CHB")
sizes <- c(5, 10, 15, 20)
reps <- 1:10
stat_cols <- c("pi", "tajima_d", "maf", "seg_sites", "h_mean", "h_std")

get_time_param <- function(pop) {
  if (pop == "YRI") "T_B" else "T_EU_AS"
}

build_marginal_cols <- function(prefix) {
  cols <- c()
  for (s in stat_cols) {
    cols <- c(cols, paste0(prefix, s, "_mean"))
    cols <- c(cols, paste0(prefix, s, "_q25"))
    cols <- c(cols, paste0(prefix, s, "_q50"))
    cols <- c(cols, paste0(prefix, s, "_q75"))
  }
  cols
}

build_corr_cols <- function(prefix) {
  cols <- c()
  n <- length(stat_cols)
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      cols <- c(cols, paste0(prefix, "corr_", stat_cols[i], "__", stat_cols[j]))
    }
  }
  cols
}

build_all_cols <- function(prefix) {
  c(build_marginal_cols(prefix), build_corr_cols(prefix))
}

feature_sets <- list(
  combined = list(dir = "abc_results", builder = build_all_cols)
)

for (fs in feature_sets) {
  dir.create(file.path(base_output, fs$dir), recursive = TRUE, showWarnings = FALSE)
}

prep_stats <- function(sim_mat_full, sim_cols) {
  missing <- setdiff(sim_cols, colnames(sim_mat_full))
  stats <- sim_mat_full[, sim_cols, drop = FALSE]
  non_const <- apply(stats, 2, function(x) mad(x) > 0)
  cat("    kept", sum(non_const), "of", length(non_const), "features\n")
  list(stats = stats[, non_const, drop = FALSE], keep = non_const)
}

run_abc <- function(prep, obs_vec, obs_cols, params, sim_ids) {
  tgt <- obs_vec[obs_cols[prep$keep]]
  if (any(is.na(tgt))) warning("NA in observed features")
  res <- abc(
    target = tgt,
    param = params,
    sumstat = prep$stats,
    tol = ABC_TOL,
    method = "rejection"
  )
  accepted_idx <- which(res$region)
  list(
    posterior = as.data.frame(res$unadj.values),
    accepted_ids = sim_ids[accepted_idx]
  )
}

load_sims <- function(path, pop) {
  t_param <- get_time_param(pop)
  tbl <- read_parquet(
    path,
    col_select = c("sim_id",
                    tidyselect::starts_with("sim_"),
                    tidyselect::all_of(c(t_param, "N0"))),
    as_data_frame = FALSE
  )
  all_sim <- setdiff(grep("^sim_", names(tbl), value = TRUE), "sim_id")
  list(
    sim_mat = as.matrix(as.data.frame(tbl[, all_sim])),
    params = as.matrix(as.data.frame(tbl[, c(t_param, "N0")])),
    sim_ids = as.data.frame(tbl[, "sim_id"])$sim_id
  )
}

load_obs <- function(path) {
  tbl <- read_parquet(
    path,
    col_select = tidyselect::starts_with("obs_"),
    as_data_frame = FALSE
  )
  obs_df <- as.data.frame(tbl$Slice(0, 1))
  obs_vec <- as.numeric(obs_df[1, ])
  names(obs_vec) <- names(obs_df)
  obs_vec
}

cat("=== ", target, " ===\n")

full_path <- file.path(csv_dir, paste0(target, "_full_v_sims.parquet"))
full_sims <- load_sims(full_path, target)
full_obs <- load_obs(full_path)
cat("  loaded full sims:", nrow(full_sims$sim_mat), "rows\n")

sub_sims <- list()
sub_obs <- list()
for (size in sizes) {
  sk <- as.character(size)
  sub_obs[[sk]] <- list()
  for (rep in reps) {
    path <- file.path(
      csv_dir,
      paste0(target, "_n", size, "_rep", rep, "_subsample_v_sims.parquet")
    )
    if (rep == reps[1]) {
      sub_sims[[sk]] <- load_sims(path, target)
      cat("  loaded n=", size, " sims:", nrow(sub_sims[[sk]]$sim_mat), "rows\n")
    }
    sub_obs[[sk]][[rep]] <- load_obs(path)
  }
}
cat("  loaded subsample obs\n")

tf_obs <- list()
for (size in sizes) {
  sk <- as.character(size)
  tf_obs[[sk]] <- list()
  for (reference in populations) {
    tf_obs[[sk]][[reference]] <- list()
    for (rep in reps) {
      path <- file.path(
        csv_dir,
        paste0(target, "_from_", reference, "_n", size, "_rep", rep,
               "_transform_v_sims.parquet")
      )
      tf_obs[[sk]][[reference]][[rep]] <- load_obs(path)
    }
  }
}
cat("  loaded transform obs\n")

for (fs_name in names(feature_sets)) {
  fs <- feature_sets[[fs_name]]
  out_dir <- file.path(base_output, fs$dir)
  cat("\n== feature set:", fs_name, "->", out_dir, "==\n")

  sim_cols <- fs$builder("sim_")
  obs_cols <- fs$builder("obs_")

  full_prep <- prep_stats(full_sims$sim_mat, sim_cols)
  full_result <- run_abc(
    full_prep, full_obs, obs_cols, full_sims$params, full_sims$sim_ids
  )
  post_full <- full_result$posterior
  full_accepted <- full_result$accepted_ids

  wd_rows <- list()
  post_rows <- list()
  overlap_rows <- list()
  acc_id_rows <- list()

  pf <- post_full
  colnames(pf) <- c("T", "N0")
  pf$target <- target
  pf$type <- "full"
  pf$reference <- NA
  pf$n <- NA
  pf$replicate <- NA
  post_rows[[length(post_rows) + 1]] <- pf

  acc_id_rows[[length(acc_id_rows) + 1]] <- data.frame(
    target = target, type = "full", reference = NA,
    n = NA, replicate = NA, sim_id = full_accepted
  )

  for (size in sizes) {
    sk <- as.character(size)
    s <- sub_sims[[sk]]
    sub_prep <- prep_stats(s$sim_mat, sim_cols)

    for (rep in reps) {
      sub_result <- run_abc(
        sub_prep, sub_obs[[sk]][[rep]], obs_cols, s$params, s$sim_ids
      )
      post_sub <- sub_result$posterior
      sub_accepted <- sub_result$accepted_ids

      wd_rows[[length(wd_rows) + 1]] <- data.frame(
        target = target, type = "subsample", reference = target,
        n = size, replicate = rep,
        wd_T = wasserstein1d(post_full[, 1], post_sub[, 1]),
        wd_N0 = wasserstein1d(post_full[, 2], post_sub[, 2])
      )

      n_overlap <- length(intersect(sub_accepted, full_accepted))
      overlap_rows[[length(overlap_rows) + 1]] <- data.frame(
        target = target, type = "subsample", reference = target,
        n = size, replicate = rep,
        n_accepted = length(sub_accepted),
        n_full_accepted = length(full_accepted),
        n_overlap = n_overlap,
        pct_overlap = n_overlap / length(sub_accepted) * 100
      )

      ps <- post_sub
      colnames(ps) <- c("T", "N0")
      ps$target <- target
      ps$type <- "subsample"
      ps$reference <- target
      ps$n <- size
      ps$replicate <- rep
      post_rows[[length(post_rows) + 1]] <- ps

      acc_id_rows[[length(acc_id_rows) + 1]] <- data.frame(
        target = target, type = "subsample", reference = target,
        n = size, replicate = rep, sim_id = sub_accepted
      )
    }

    for (reference in populations) {
      for (rep in reps) {
        tf_result <- run_abc(
          full_prep, tf_obs[[sk]][[reference]][[rep]], obs_cols,
          full_sims$params, full_sims$sim_ids
        )
        post_tf <- tf_result$posterior
        tf_accepted <- tf_result$accepted_ids

        wd_rows[[length(wd_rows) + 1]] <- data.frame(
          target = target, type = "transform", reference = reference,
          n = size, replicate = rep,
          wd_T  = wasserstein1d(post_full[, 1], post_tf[, 1]),
          wd_N0 = wasserstein1d(post_full[, 2], post_tf[, 2])
        )

        n_overlap <- length(intersect(tf_accepted, full_accepted))
        overlap_rows[[length(overlap_rows) + 1]] <- data.frame(
          target = target, type = "transform", reference = reference,
          n = size, replicate = rep,
          n_accepted = length(tf_accepted),
          n_full_accepted = length(full_accepted),
          n_overlap = n_overlap,
          pct_overlap = n_overlap / length(tf_accepted) * 100
        )

        pt <- post_tf
        colnames(pt) <- c("T", "N0")
        pt$target <- target
        pt$type <- "transform"
        pt$reference <- reference
        pt$n <- size
        pt$replicate <- rep
        post_rows[[length(post_rows) + 1]] <- pt

        acc_id_rows[[length(acc_id_rows) + 1]] <- data.frame(
          target = target, type = "transform", reference = reference,
          n = size, replicate = rep, sim_id = tf_accepted
        )
      }
    }
  }

  write.csv(
    do.call(rbind, wd_rows),
    file.path(out_dir, paste0("wd_raw_", target, ".csv")),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, post_rows),
    file.path(out_dir, paste0("posteriors_", target, ".csv")),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, overlap_rows),
    file.path(out_dir, paste0("overlaps_", target, ".csv")),
    row.names = FALSE
  )
  write.csv(
    do.call(rbind, acc_id_rows),
    file.path(out_dir, paste0("accepted_ids_", target, ".csv")),
    row.names = FALSE
  )
  cat("  wrote", fs_name, "results\n")
}

cat("Done.\n")