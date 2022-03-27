package Interface.ML;

import com.opencsv.exceptions.CsvValidationException;
import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.math.optimisers.AdaGrad;

import java.io.IOException;
import static io.vavr.API.println;

public class LogisticRegressionClassifier extends CSVData implements AllProcesses {

    private final Trainer<Label> LRTrainer;
    private Model<Label> Learner;
    private DataSource<Label> TrainData;
    private DataSource<Label> TestData;
    private MutableDataset<Label> TrainingPart;
    private MutableDataset<Label> TestingPart;
    private LabelEvaluation Evaluator;

    public LogisticRegressionClassifier(int NumEpochs) {
        LRTrainer = new LinearSGDTrainer(new LogMulticlass(), new AdaGrad(0.1), NumEpochs, Trainer.DEFAULT_SEED);
    }

    protected void ReadData(String DataHeaderPath, String TrainDataPath, String TestDataPath) throws IOException, CsvValidationException {
        ReadDataFromCSV(DataHeaderPath, TrainDataPath, TestDataPath);
        TrainData = GetTrainData();
        TestData = GetTestData();
    }

    @Override
    public void GenerateDataSource() {
        TrainingPart = new MutableDataset<>(TrainData);
        TestingPart = new MutableDataset<>(TestData);
    }

    @Override
    public void TrainClassifier() {
        Learner = LRTrainer.train(TrainingPart);
    }

    @Override
    public void EvaluateClassifier() {
        Evaluator = new LabelEvaluator().evaluate(Learner, TestingPart);
    }

    @Override
    public void DisplaySummary() {
        println("\nThe performance of LR Classifier is : \n" +  Evaluator + "\nThe confusion matrix is : \n" +
                Evaluator.getConfusionMatrix());
    }

    protected void SaveResult(String SavePath) throws IOException {
        SaveResults(SavePath, Evaluator);
    }

    @Override
    public double GetAccuracy() {
        return Evaluator.accuracy();
    }
}
