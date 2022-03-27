package Interface.ML;

import com.opencsv.exceptions.CsvValidationException;
import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.math.optimisers.AdaGrad;

import java.io.IOException;

import static io.vavr.API.println;


public class FactorizationMachines extends CSVData implements AllProcesses {

    private final Trainer<Label> FMTrainer;
    private Model<Label> Learner;
    private DataSource<Label> TrainData;
    private DataSource<Label> TestData;
    private MutableDataset<Label> TrainPart;
    private MutableDataset<Label> TestPart;
    private LabelEvaluation Evaluator;

    public FactorizationMachines(int Epochs) {
        FMTrainer = new FMClassificationTrainer(new LogMulticlass(), new AdaGrad(0.1f), Epochs, Trainer.DEFAULT_SEED, 6, 0.1d);
    }

    protected void ReadData(String DataHeaderPath, String TrainDataPath, String TestDataPath) throws CsvValidationException, IOException {
        ReadDataFromCSV(DataHeaderPath, TrainDataPath, TestDataPath);
        TrainData = GetTrainData();
        TestData = GetTestData();
    }

    @Override
    public void GenerateDataSource() {
        TrainPart = new MutableDataset<>(TrainData);
        TestPart = new MutableDataset<>(TestData);
    }

    @Override
    public void TrainClassifier() {
        Learner = FMTrainer.train(TrainPart);
    }

    @Override
    public void EvaluateClassifier() {
        Evaluator = new LabelEvaluator().evaluate(Learner, TestPart);
    }

    @Override
    public void DisplaySummary() {
        println("\nThe performance of FM Classifier is : \n" +  Evaluator + "\nThe confusion matrix is : \n" +
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
