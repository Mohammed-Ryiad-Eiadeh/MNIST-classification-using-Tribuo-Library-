package MnisClassification;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.liblinear.LibLinearClassificationTrainer;
import org.tribuo.classification.mnb.MultinomialNaiveBayesTrainer;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.kernel.KernelSVMTrainer;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.common.nearest.KNNClassifierOptions;
import org.tribuo.common.nearest.KNNTrainer.Distance;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.optimisers.AdaGrad;

import java.io.*;
import java.nio.file.Paths;

import static io.vavr.API.println;

public class MainClass {
    public static void main(String[] args) throws IOException {

        var AttName = new BufferedReader(new FileReader(new File("C:\\Users\\LENOVO\\IdeaProjects\\MNIST_CSV_Classification\\mnist_header.csv")));
        var header = AttName.readLine().split(",");

        var TrainData = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get("C:\\Users\\LENOVO\\IdeaProjects\\MNIST_CSV_Classification\\mnist_train.csv")
                ,header[header.length - 1]
                ,header);

        var TestData = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get("C:\\Users\\LENOVO\\IdeaProjects\\MNIST_CSV_Classification\\mnist_test.csv")
                ,header[header.length - 1]
                ,header);

        var train = new MutableDataset<>(TrainData);
        var test = new MutableDataset<>(TestData);

        println(String.format("train data size = %d, number of features = %d, number of labels = %d", train.size(),
                train.getFeatureMap().size(),
                train.getOutputInfo().size()));

        println(String.format("test data size = %d, number of features = %d, number of labels = %d", test.size(),
                test.getFeatureMap().size(),
                test.getOutputInfo().size()));

        LabelEvaluation Evaluator;

        var FactorizationMachine = new FMClassificationTrainer(new Hinge(), new AdaGrad(0.1), 1, 12345L, 6, 0.1);
        var FMLearner = FactorizationMachine.train(train);
        Evaluator = new LabelEvaluator().evaluate(FMLearner, test);
        println("----------------Factorization Machine Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var SVMTrainer = new KernelSVMTrainer(new Linear(), 0.1, 1, Trainer.DEFAULT_SEED);
        var SVMLearner = SVMTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(SVMLearner, test);
        println("----------------Support Vector Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var LRTrainer = new LogisticRegressionTrainer();
        var LRLearner = LRTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(LRLearner, test);
        println("----------------Logistic Regression Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var MBayesTrainer = new MultinomialNaiveBayesTrainer();
        var MBLearner = MBayesTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(MBLearner, test);
        println("----------------MBayes Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var KNNTrainer = new KNNClassifierOptions();
        KNNTrainer.knnK = 3; KNNTrainer.knnDistance = Distance.L1;
        var KNNLearner = KNNTrainer.getTrainer().train(train);
        Evaluator = new LabelEvaluator().evaluate(KNNLearner, test);
        println("----------------KNN Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var CartTrainer = new CARTClassificationTrainer();
        var CartLearner = CartTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(CartLearner, test);
        println("----------------CART Tree Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var LibLinTrainer = new LibLinearClassificationTrainer();
        var LibLinLearner = LibLinTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(LibLinLearner, test);
        println("----------------LibLinear Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");

        var XGBTrainer = new XGBoostClassificationTrainer(10);
        var XGBLearner = XGBTrainer.train(train);
        Evaluator = new LabelEvaluator().evaluate(XGBLearner, test);
        println("----------------XGBoost Performance\n" + Evaluator + "\n----------------Confusion Matrix \n" + Evaluator.getConfusionMatrix() + "\n\n");
    }
}
