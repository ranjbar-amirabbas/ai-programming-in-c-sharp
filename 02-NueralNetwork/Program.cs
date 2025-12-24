using _02_NueralNetwork;

var neuralNetwork = new NeuralNetwork();

double[,] trainingSetInputs = new double[,]
{
    { 0, 0, 0 },
    { 1, 1, 1 },
    { 1, 0, 0 }
};

double[,] trainingSetOutputs = new double[,]
{
    { 0 },
    { 1 },
    { 1 }
};

neuralNetwork.Train(trainingSetInputs, trainingSetOutputs, 1000);

double[,] output = neuralNetwork.Think(
    new double[,]
    {
        { 0, 1, 0 },
        { 0, 0, 0 },
        { 0, 0, 1 }
    }
);

PrintMatrix(output);

static void PrintMatrix(double[,] matrix)
{
    int rows = matrix.GetLength(0);
    int columns = matrix.GetLength(1);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            Console.Write(Math.Round(matrix[row, column]) + " ");
        }

        Console.WriteLine();
    }
}