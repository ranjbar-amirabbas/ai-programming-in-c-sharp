using Microsoft.ML.Data;

namespace _03_ML;


// ----------------------------------------------------
// Data input schema (matches CSV columns)
// ----------------------------------------------------
public class HousingData
{
    [LoadColumn(0)]
    public float SquareFeet { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    // Label column (what the model learns to predict)
    [LoadColumn(2)]
    public float Price { get; set; }
}

// ----------------------------------------------------
// Model output schema
// ----------------------------------------------------
public class HousingPrediction
{
    // Regression predictions are stored in "Score"
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}
