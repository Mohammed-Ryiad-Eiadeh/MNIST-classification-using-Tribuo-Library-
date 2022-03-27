package Interface.ML;

import com.opencsv.exceptions.CsvValidationException;
import java.io.IOException;

public class MainClass {
    public static void main(String ...args) throws CsvValidationException, IOException {

        var Header = "C:\\Users\\LENOVO\\IdeaProjects\\MNIST_CSV_Classification\\mnist_header.csv";
        var TrainData = "C:\\Users\\LENOVO\\IdeaProjects\\MNIST_CSV_Classification\\mnist_train.csv";
        var TestingData = "C:\\Users\\LENOVO\\IdeaProjects\\MNIST_CSV_Classification\\mnist_test.csv";

        var Accuracy = new double[3];

        var LRTrainer = new LogisticRegressionClassifier(1);
        LRTrainer.ReadData(Header, TrainData, TestingData);
        LRTrainer.GenerateDataSource();
        LRTrainer.TrainClassifier();
        LRTrainer.EvaluateClassifier();
        LRTrainer.DisplaySummary();
        Accuracy[0] = LRTrainer.GetAccuracy();
        LRTrainer.SaveResult("C:\\Users\\LENOVO\\Desktop\\LR-Result.csv");

        var XGBoostTrainer = new XGBoostClassifier(1);
        XGBoostTrainer.ReadData(Header, TrainData, TestingData);
        XGBoostTrainer.GenerateDataSource();
        XGBoostTrainer.TrainClassifier();
        XGBoostTrainer.EvaluateClassifier();
        XGBoostTrainer.DisplaySummary();
        Accuracy[1] = XGBoostTrainer.GetAccuracy();
        XGBoostTrainer.SaveResult("C:\\Users\\LENOVO\\Desktop\\XgBoost-Result.csv");

        var FMTRainer = new FactorizationMachines(1);
        FMTRainer.ReadData(Header, TrainData, TestingData);
        FMTRainer.GenerateDataSource();
        FMTRainer.TrainClassifier();
        FMTRainer.EvaluateClassifier();
        FMTRainer.DisplaySummary();
        Accuracy[2] = FMTRainer.GetAccuracy();
        FMTRainer.SaveResult("C:\\Users\\LENOVO\\Desktop\\FM-Result.csv");

        var XTitle = "LR                                      XgBoost                                  FM";
        var chart = new Plotting("performance", XTitle, "Accuracy", 450, 450, ChartTheme.MATLAB);
        chart.SetValues(Accuracy);
        chart.Display();
    }
}
