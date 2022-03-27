package Interface.ML;

import com.opencsv.exceptions.CsvValidationException;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;

import java.io.IOException;

import static io.vavr.API.println;

public class XGBoostClassifier extends CSVData implements AllProcesses {

    private final Trainer<Label> trainer;
    private Model<Label> Learner;
    private DataSource<Label> TrainData;
    private DataSource<Label> TestData;
    private MutableDataset<Label> TrainingPart;
    private MutableDataset<Label> TestingPart;
    private LabelEvaluation Evaluator;

    public XGBoostClassifier(int NumTrees) {
        trainer = new XGBoostClassificationTrainer(NumTrees);
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
        Learner = trainer.train(TrainingPart);
    }

    @Override
    public void EvaluateClassifier() {
        Evaluator = new LabelEvaluator().evaluate(Learner, TestingPart);
    }

    @Override
    public void DisplaySummary() {
        println("\nThe performance of XgBoost Classifier is : \n" +  Evaluator + "\nThe confusion matrix is : \n" +
                Evaluator.getConfusionMatrix());
    }

    public void SaveResult(String SavePath) throws IOException {
        SaveResults(SavePath, Evaluator);
    }

    @Override
    public double GetAccuracy() {
        return Evaluator.accuracy();
    }
}
