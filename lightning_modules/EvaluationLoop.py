from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
import os
import shutil
import sys
from typing import Any, List

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor

from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT
from lightning.pytorch.trainer import call

class customEvaluationLoop(_EvaluationLoop):
    def _on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_end`` hook."""
        trainer = self.trainer
        hook_name = "on_test_end" if trainer.testing else "on_validation_end"
        call._call_lightning_module_hook(trainer, hook_name, *args, **kwargs)
        call._call_callback_hooks(trainer, hook_name, *args, **kwargs)
        call._call_strategy_hook(trainer, hook_name, *args, **kwargs)

        # reset the logger connector state
        trainer._logger_connector.reset_results()

    @staticmethod
    def _print_results(results: List[_OUT_DICT], stage: str) -> None:
        # remove the dl idx suffix
        headers = [list(result.keys())[-2].split("/")[-2] for result in results]
        results = [{k.split("/")[-1]: v for k, v in result.items()} for result in results]
        metrics_paths = {k for keys in apply_to_collection(results, dict, _EvaluationLoop._get_keys) for k in keys}
        if not metrics_paths:
            return

        metrics_strs = [":".join(metric) for metric in metrics_paths]
        # sort both lists based on metrics_strs
        metrics_strs, metrics_paths = zip(*sorted(zip(metrics_strs, metrics_paths)))

        # fallback is useful for testing of printed output
        term_size = shutil.get_terminal_size(fallback=(120, 30)).columns or 120
        max_length = int(min(max(len(max(metrics_strs, key=len)), len(max(headers, key=len)), 25), term_size / 2))

        rows: List[List[Any]] = [[] for _ in metrics_paths]

        for result in results:
            for metric, row in zip(metrics_paths, rows):
                val = _EvaluationLoop._find_value(result, metric)
                if val is not None:
                    if isinstance(val, Tensor):
                        val = val.item() if val.numel() == 1 else val.tolist()
                    # row.append(f"{val}")
                    row.append("{:.8f}".format(val))
                else:
                    row.append(" ")

        # keep one column with max length for metrics
        num_cols = int((term_size - max_length) / max_length)

        for i in range(0, len(headers), num_cols):
            table_headers = headers[i : (i + num_cols)]
            table_rows = [row[i : (i + num_cols)] for row in rows]

            table_headers.insert(0, f"{stage} Metric".capitalize())

            if _RICH_AVAILABLE:
                from rich import get_console
                from rich.table import Column, Table

                columns = [Column(h, justify="center", style="magenta", width=max_length) for h in table_headers]
                columns[0].style = "cyan"

                table = Table(*columns)
                for metric, row in zip(metrics_strs, table_rows):
                    row.insert(0, metric)
                    table.add_row(*row)

                console = get_console()
                console.print(table)
            else:
                row_format = f"{{:^{max_length}}}" * len(table_headers)
                half_term_size = int(term_size / 2)

                try:
                    # some terminals do not support this character
                    if sys.stdout.encoding is not None:
                        "─".encode(sys.stdout.encoding)
                except UnicodeEncodeError:
                    bar_character = "-"
                else:
                    bar_character = "─"
                bar = bar_character * term_size

                lines = [bar, row_format.format(*table_headers).rstrip(), bar]
                for metric, row in zip(metrics_strs, table_rows):
                    # deal with column overflow
                    if len(metric) > half_term_size:
                        while len(metric) > half_term_size:
                            row_metric = metric[:half_term_size]
                            metric = metric[half_term_size:]
                            lines.append(row_format.format(row_metric, *row).rstrip())
                        lines.append(row_format.format(metric, " ").rstrip())
                    else:
                        lines.append(row_format.format(metric, *row).rstrip())
                lines.append(bar)
                print(os.linesep.join(lines))