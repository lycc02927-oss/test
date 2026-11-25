"""从已有 CSV 生成 Excel。可以脱离 GA 重新导出，不用重新跑寻优。"""
import argparse
import csv
from pathlib import Path

TABS = {
    "selection": "入选配方", "overall": "配方总评", "summary": "工况汇总", "trials": "逐次试验",
    "points": "采样轨迹", "recipes": "全部配方", "generations": "代际进展",
    "members": "每代成员", "models": "模型参考", "settings": "试验设置",
    "seeds": "随机种子", "grid": "编码档位",
}
TEXT = {"role", "stage", "phase", "stop", "chromosome", "parameter", "name", "plan", "distance_metric"}
PERCENT = {"error", "mean_error", "error_limit", "success_rate", "worst_failure_rate",
           "mean_failure_rate", "failure_rate", "failure_weight", "crossover", "bit_mutation"}


def parse(key, value):
    if not value:
        return None
    if key in TEXT:
        return value
    # 初始点数、试验编号保留整数类型；ED 和偏差保留浮点精度。
    return float(value) if "." in value or "e" in value.lower() else int(value)


def column(index):
    label = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def sheet_rows(directory, key):
    """迭代产出表头和行；尽量避免在导出时多存一份完整采样轨迹。"""
    if key == "trials":
        with (directory / "settings.csv").open(encoding="utf-8-sig", newline="") as config:
            # 老批次的 CSV 未声明计分距离，重导历史工作簿时保留其原始公式。
            distance_metric = next(csv.DictReader(config)).get("distance_metric", "travel")
    with (directory / (key + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        if key == "trials":
            axis_label = "max_axis_total_check" if distance_metric == "sum_step_max" else "travel_check"
            yield headers + ["error_check", axis_label, "loss_check"]
        else:
            yield headers
        col = {h: column(i) for i, h in enumerate(headers, 1)}
        for n, row in enumerate(reader, 2):
            values = [parse(k, v) for k, v in zip(headers, row)]
            if key == "trials":
                ed = col["best_ed"] + str(n)
                peak = col["design_ed"] + str(n)
                sample = col["samples"] + str(n)
                axes = [col["travel_" + a] + str(n) for a in "xyz"]
                error_check = column(len(headers) + 1) + str(n)
                cost_distance = col[distance_metric] + str(n)
                # 试验设置 F/G/I 为三个归一化分母，J/K/L 为权重。
                settings = "'试验设置'!"
                score = (f"=({error_check}/{settings}$F$2*{settings}$J$2"
                         f"+{sample}/{settings}$G$2*{settings}$K$2"
                         f"+{cost_distance}/{settings}$I$2*{settings}$L$2)"
                         f"/SUM({settings}$J$2:$L$2)")
                if "patience_failed" in col:
                    # 新批次 N2 是失败率权重；历史批次保留其原始评分公式。
                    weight = f"{settings}$N$2"
                    failed = col["patience_failed"] + str(n)
                    score = f"=(1-{weight})*({score[1:]})+{weight}*{failed}"
                values += [f"=ABS({ed}-{peak})/ABS({peak})", f"=MAX({','.join(axes)})", score]
            yield values


def export(directory):
    # 使用 openpyxl 导出。原始 CSV 始终保留。
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.chart import LineChart, Reference
    from openpyxl.styles import Font, PatternFill

    # standard 会有较多轨迹行，流式写入使内存占用与行数无关。
    workbook = Workbook(write_only=True)
    for key, title in TABS.items():
        sheet = workbook.create_sheet(title)
        rows = sheet_rows(directory, key)
        headers = next(rows)
        sheet.freeze_panes = "A2"
        styled_headers = []
        for index, header in enumerate(headers, 1):
            width = 54 if header == "chromosome" else max(14, len(header) + 3)
            sheet.column_dimensions[column(index)].width = width
            cell = WriteOnlyCell(sheet, value=header)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="224863")
            styled_headers.append(cell)
        sheet.append(styled_headers)
        percentages = [i for i, h in enumerate(headers) if h in PERCENT or h == "error_check"]
        last_row = 1
        for last_row, row in enumerate(rows, 2):
            for i in percentages:
                cell = WriteOnlyCell(sheet, value=row[i])
                cell.number_format = "0.00%"
                row[i] = cell
            sheet.append(row)
        sheet.auto_filter.ref = f"A1:{column(len(headers))}{last_row}"
        if key == "generations" and "population_mean_loss" in headers:
            chart = LineChart()
            chart.title = "GA 训练 loss（第 0 代为默认 BO）"
            chart.x_axis.title, chart.y_axis.title = "代数", "综合 loss"
            for field in ("mean_loss", "population_mean_loss"):
                idx = headers.index(field) + 1
                chart.add_data(Reference(sheet, min_col=idx, min_row=1, max_row=last_row),
                               titles_from_data=True)
            chart.set_categories(Reference(sheet, min_col=1, min_row=2, max_row=last_row))
            sheet.add_chart(chart, f"{column(len(headers) + 2)}2")
    workbook.save(directory / "GA_results.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本次实验的 CSV 重新生成工作簿")
    parser.add_argument("directory", type=Path)
    export(parser.parse_args().directory)
