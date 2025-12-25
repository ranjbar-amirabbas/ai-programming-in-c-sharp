using System.Globalization;
using _03_ML;
using Microsoft.ML;

// ----------------------------------------------------
// 1) Create ML.NET context (entry point for all ML ops)
// ----------------------------------------------------
var mlContext = new MLContext(seed: 1);

// ----------------------------------------------------
// 2) Load CSV data from disk
// ----------------------------------------------------
var dataPath = Path.Combine(AppContext.BaseDirectory, "housing-data.csv");

var data = mlContext.Data.LoadFromTextFile<HousingData>(
    path: dataPath,
    separatorChar: ',',
    hasHeader: true);

// ----------------------------------------------------
// 3) Sanity check: print first few rows to ensure
//    data is loaded and mapped correctly
// ----------------------------------------------------
Console.WriteLine("---- Loaded rows ----");
foreach (var row in mlContext.Data
    .CreateEnumerable<HousingData>(data, reuseRowObject: false)
    .Take(4))
{
    Console.WriteLine(
        $"sqft={row.SquareFeet}, beds={row.Bedrooms}, price={row.Price}");
}
Console.WriteLine("---------------------");

// ----------------------------------------------------
// 4) Split data into Train / Test sets
//    (important even for small datasets)
// ----------------------------------------------------
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.25);

// ----------------------------------------------------
// 5) Build training pipeline
//    - Combine feature columns into a single vector
//    - Normalize feature values
//    - Train a regression model (SDCA)
// ----------------------------------------------------
var pipeline =
    mlContext.Transforms.Concatenate(
        "Features",
        nameof(HousingData.SquareFeet),
        nameof(HousingData.Bedrooms))
    .Append(
        mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(
        mlContext.Regression.Trainers.Sdca(
            labelColumnName: nameof(HousingData.Price),
            featureColumnName: "Features"));

// ----------------------------------------------------
// 6) Train the model using training data
// ----------------------------------------------------
var model = pipeline.Fit(split.TrainSet);

// ----------------------------------------------------
// 7) Evaluate model performance using test data
// ----------------------------------------------------
var testPredictions = model.Transform(split.TestSet);

var metrics = mlContext.Regression.Evaluate(
    testPredictions,
    labelColumnName: nameof(HousingData.Price));

Console.WriteLine($"MAE  (Mean Absolute Error): {metrics.MeanAbsoluteError}");
Console.WriteLine($"RMSE (Root Mean Squared Error): {metrics.RootMeanSquaredError}");
Console.WriteLine("--------------------------------------------------");

// ----------------------------------------------------
// 8) Create PredictionEngine for single predictions
//    (good for console apps / demos)
// ----------------------------------------------------
var predictor =
    mlContext.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

// ----------------------------------------------------
// 9) Sanity prediction to ensure model works
// ----------------------------------------------------
var sanityPrediction = predictor.Predict(
    new HousingData { SquareFeet = 1500, Bedrooms = 3 });

Console.WriteLine(
    $"Sanity prediction (1500 sqft, 3 beds): {sanityPrediction.PredictedPrice}");
Console.WriteLine("--------------------------------------------------");

// ----------------------------------------------------
// 10) Helper method to safely parse floats using
//     invariant culture (avoids locale issues)
// ----------------------------------------------------
static bool TryParseFloatInvariant(string s, out float value)
{
    return float.TryParse(
        s,
        NumberStyles.Float,
        CultureInfo.InvariantCulture,
        out value);
}

// ----------------------------------------------------
// 11) Interactive loop: read user input and predict
// ----------------------------------------------------
while (true)
{
    Console.Write("Enter SquareFeet (or 'q' to quit): ");
    var sqftText = Console.ReadLine();

    if (string.Equals(sqftText, "q", StringComparison.OrdinalIgnoreCase))
        break;

    Console.Write("Enter Bedrooms: ");
    var bedsText = Console.ReadLine();

    if (!TryParseFloatInvariant(sqftText!, out var sqft) ||
        !TryParseFloatInvariant(bedsText!, out var beds))
    {
        Console.WriteLine("Invalid input. Please enter numeric values.\n");
        continue;
    }

    var prediction = predictor.Predict(
        new HousingData
        {
            SquareFeet = sqft,
            Bedrooms = beds
        });

    Console.WriteLine($"Predicted Price: {prediction.PredictedPrice}\n");
}
