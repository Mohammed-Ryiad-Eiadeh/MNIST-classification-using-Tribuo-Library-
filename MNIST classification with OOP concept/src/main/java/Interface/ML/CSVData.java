package Interface.ML;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvValidationException;
import org.tribuo.DataSource;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.data.csv.CSVLoader;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

public class CSVData {

    private DataSource<Label> TrainData;
    private DataSource<Label> TestData;

    protected void ReadDataFromCSV(String DataHeaderPath, String TrainDataPath, String TestDataPath) throws IOException, CsvValidationException {
        var Reader = new CSVReader(new FileReader(DataHeaderPath));
        var Header = Reader.readNext();

        TrainData = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(TrainDataPath),
                Header[Header.length - 1],
                Header);

        TestData = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(TestDataPath),
                Header[Header.length - 1],
                Header);
    }

    protected DataSource<Label> GetTrainData() {
        return TrainData;
    }

    protected DataSource<Label> GetTestData() {
        return TestData;
    }

    protected void SaveResults(String SavePath, LabelEvaluation Evaluator) throws IOException {
        var CSVWriter = new CSVWriter(new FileWriter(SavePath));
        String[][] Data = {{"TP" , "FP", "TN", "FN", "Accuracy"},
                {String.valueOf(Evaluator.tp()),
                        String.valueOf(Evaluator.fp()),
                        String.valueOf(Evaluator.tn()),
                        String.valueOf(Evaluator.fn()),
                        String.valueOf(Evaluator.accuracy())}};
        CSVWriter.writeAll(Arrays.asList(Data));
        CSVWriter.flush();
        CSVWriter.close();
    }
}
