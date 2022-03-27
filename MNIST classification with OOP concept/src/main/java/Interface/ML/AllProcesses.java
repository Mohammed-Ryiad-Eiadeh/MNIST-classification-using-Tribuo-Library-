package Interface.ML;

public interface AllProcesses {
    void GenerateDataSource();
    void TrainClassifier();
    void EvaluateClassifier();
    void DisplaySummary();
    double GetAccuracy();
}
