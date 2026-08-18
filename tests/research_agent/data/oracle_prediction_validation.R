# Independent base-R oracle for CAP-PREDVAL-V2.
#
# Regenerate the frozen JSON with:
#   Rscript tests/research_agent/data/oracle_prediction_validation.R \
#     tests/research_agent/data/oracle_prediction_validation.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("expected exactly one prediction-validation CSV path")
}

frame <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
required <- c("split", "outcome", "probability")
if (!all(required %in% names(frame))) {
  stop("oracle fixture is missing required columns")
}
evaluation <- frame[trimws(as.character(frame$split)) == "test", , drop = FALSE]
outcome <- as.integer(evaluation$outcome)
probability <- as.numeric(evaluation$probability)
if (
  nrow(evaluation) == 0 ||
  any(is.na(outcome)) ||
  any(is.na(probability)) ||
  !identical(sort(unique(outcome)), c(0L, 1L)) ||
  any(probability < 0 | probability > 1)
) {
  stop("oracle evaluation rows are invalid")
}

event_n <- sum(outcome == 1L)
non_event_n <- sum(outcome == 0L)
ranks <- rank(probability, ties.method = "average")
auroc <- (
  sum(ranks[outcome == 1L]) - event_n * (event_n + 1) / 2
) / (event_n * non_event_n)
brier_score <- mean((outcome - probability)^2)
logit_probability <- qlogis(probability)
calibration <- glm(
  outcome ~ logit_probability,
  family = binomial(link = "logit")
)

threshold_counts <- function(threshold) {
  positive <- probability >= threshold
  c(
    tp = sum(positive & outcome == 1L),
    fp = sum(positive & outcome == 0L),
    tn = sum(!positive & outcome == 0L),
    fn = sum(!positive & outcome == 1L)
  )
}

render_number <- function(value) {
  formatC(value, digits = 17, format = "g")
}

render_counts <- function(values) {
  sprintf(
    '{"tp":%d,"fp":%d,"tn":%d,"fn":%d}',
    values[["tp"]],
    values[["fp"]],
    values[["tn"]],
    values[["fn"]]
  )
}

coefficients <- unname(coef(calibration))
cat(
  paste0(
    "{",
    '"evaluation_n":', nrow(evaluation), ",",
    '"auroc":', render_number(auroc), ",",
    '"brier_score":', render_number(brier_score), ",",
    '"calibration_intercept":', render_number(coefficients[[1]]), ",",
    '"calibration_slope":', render_number(coefficients[[2]]), ",",
    '"threshold_metrics":{',
    '"0.5":', render_counts(threshold_counts(0.5)), ",",
    '"0.8":', render_counts(threshold_counts(0.8)),
    "}",
    "}\n"
  )
)
