from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.palettes import Colorblind
from bokeh.plotting import figure, show


def plot_loss_and_accuracies(
    losses: dict,
    train_accuracies: dict,
    val_accuracies: dict,
    notebook: bool = False,
) -> None:
    n_epochs = max([len(x) for x in list(losses.values())])
    epochs = list(range(1, n_epochs + 1))

    loss_plot = figure(
        title="Loss on Training Set per Epoch",
        x_axis_label="Epoch",
        y_axis_label="Training Loss",
        width=600,
        height=400,
        background_fill_color="#f2f2f2",
        tools="",
    )
    train_acc_plot = figure(
        title="Accuracy on Training Set per Epoch",
        x_axis_label="Epoch",
        y_axis_label="Train accuracy",
        width=600,
        height=400,
        background_fill_color="#f2f2f2",
        tools="",
        y_range=(0, 1.05),
    )
    val_acc_plot = figure(
        title="Accuracy on Validation Set per Epoch",
        x_axis_label="Epoch",
        y_axis_label="Validation Accuracy",
        width=600,
        height=400,
        background_fill_color="#f2f2f2",
        tools="",
        y_range=(0, 1.05),
    )

    colors = Colorblind[3]
    for model in losses:
        line_style = "solid" if "Cora" in model else "dashed"
        color = colors[1] if "GR" in model else colors[0]

        loss_plot.line(
            epochs,
            losses[model],
            legend_label=model,
            line_width=2,
            color=color,
            line_dash=line_style,
        )
        train_acc_plot.line(
            epochs,
            train_accuracies[model],
            legend_label=model,
            line_width=2,
            color=color,
            line_dash=line_style,
        )
        val_acc_plot.line(
            epochs,
            val_accuracies[model],
            legend_label=model,
            line_width=2,
            color=color,
            line_dash=line_style,
        )

    # Grid and axis styling
    for p in [loss_plot, train_acc_plot, val_acc_plot]:
        p.grid.grid_line_color = "white"
        p.axis.axis_line_color = None

        p.title.text_font_size = "16pt"
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.legend.label_text_font_size = "12pt"

    show(column(loss_plot, train_acc_plot, val_acc_plot))

    if notebook:
        output_notebook()  # best_acc = 0.572, test_acc = 0.558,
