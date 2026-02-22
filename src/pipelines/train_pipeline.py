import argparse

from src.models.training import TrainingConfig, train_model
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import load_yaml, resolve_path


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AraBART morphology model.")
    parser.add_argument("--config", default=str(resolve_path("src/config/default.yaml")))

    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--pretrained-model-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--trained-model-dir")

    parser.add_argument("--max-input-length", type=int)
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--seed", type=int)

    return parser


def pick(cli_value, cfg_value):
    return cli_value if cli_value is not None else cfg_value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging()
    cfg = load_yaml(args.config)

    paths_cfg = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    project_cfg = cfg.get("project", {})

    config = TrainingConfig(
        train_csv=pick(args.train_csv, paths_cfg.get("train_csv", "data_ready/train.csv")),
        val_csv=pick(args.val_csv, paths_cfg.get("val_csv", "data_ready/val.csv")),
        pretrained_model_name=pick(
            args.pretrained_model_name, model_cfg.get("pretrained_name", "moussaKam/AraBART")
        ),
        output_dir=pick(args.output_dir, paths_cfg.get("training_output_dir", "arabart_morph_model")),
        trained_model_dir=pick(
            args.trained_model_dir, paths_cfg.get("trained_model_dir", "trained_arabart_morph_model")
        ),
        max_input_length=pick(args.max_input_length, model_cfg.get("max_input_length", 64)),
        max_target_length=pick(args.max_target_length, model_cfg.get("max_target_length", 128)),
        learning_rate=pick(args.learning_rate, train_cfg.get("learning_rate", 3e-5)),
        warmup_steps=pick(args.warmup_steps, train_cfg.get("warmup_steps", 300)),
        train_batch_size=pick(args.train_batch_size, train_cfg.get("train_batch_size", 16)),
        eval_batch_size=pick(args.eval_batch_size, train_cfg.get("eval_batch_size", 2)),
        weight_decay=pick(args.weight_decay, train_cfg.get("weight_decay", 0.01)),
        num_train_epochs=pick(args.num_train_epochs, train_cfg.get("num_train_epochs", 6)),
        save_total_limit=pick(args.save_total_limit, train_cfg.get("save_total_limit", 2)),
        early_stopping_patience=pick(
            args.early_stopping_patience, train_cfg.get("early_stopping_patience", 2)
        ),
        logging_steps=pick(args.logging_steps, train_cfg.get("logging_steps", 20)),
        seed=pick(args.seed, project_cfg.get("seed", 42)),
    )

    metrics = train_model(config)
    LOGGER.info("Final evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()

