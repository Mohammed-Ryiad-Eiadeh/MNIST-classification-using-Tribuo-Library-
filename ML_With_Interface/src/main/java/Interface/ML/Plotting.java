package Interface.ML;

import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.CategorySeries;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.style.Styler;

public class Plotting {

    private CategoryChart Chart;

    public Plotting(String Title, String XTitle, String YTitle, int Width, int Height, ChartTheme Theme) {

        switch (Theme) {
            case XCHART -> Chart = new CategoryChartBuilder()
                    .title(Title).width(Width)
                    .height(Height)
                    .xAxisTitle(XTitle)
                    .yAxisTitle(YTitle)
                    .theme(Styler.ChartTheme.XChart)
                    .build();
            case GGPLOT2 -> Chart = new CategoryChartBuilder()
                    .title(Title).width(Width)
                    .height(Height)
                    .xAxisTitle(XTitle)
                    .yAxisTitle(YTitle)
                    .theme(Styler.ChartTheme.GGPlot2)
                    .build();
            case MATLAB -> Chart = new CategoryChartBuilder()
                    .title(Title).width(Width)
                    .height(Height)
                    .xAxisTitle(XTitle)
                    .yAxisTitle(YTitle)
                    .theme(Styler.ChartTheme.Matlab)
                    .build();
        }

        Chart.getStyler().setHasAnnotations(true);
        Chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        Chart.getStyler().setDefaultSeriesRenderStyle(CategorySeries.CategorySeriesRenderStyle.Bar);
    }

    protected void SetValues(double[] values) {
        Chart.addSeries("Values", new double[] {1, 2, 3}, values);
    }

    protected void Display() {
        new SwingWrapper<>(Chart).displayChart().setTitle("Performance Result");
    }
}
