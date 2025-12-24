namespace _02_NueralNetwork;

public class NeuralNetwork
{
    private double[,] _weights;

    private enum OPERATION
    {
        Multiply,
        Add,
        Subtract
    }

    public NeuralNetwork()
    {
        Random randomNumber = new Random(1);
        int numberOfInputNodes = 3;
        int numberOfOutputNodes = 1;

        _weights = new double[numberOfInputNodes, numberOfOutputNodes];

        for (int i = 0; i < numberOfInputNodes; i++)
        {
            for (int j = 0; j < numberOfOutputNodes; j++)
            {
                _weights[i, j] = 2 * randomNumber.NextDouble() - 1;
            }
        }
    }


    private double[,] Activate(double[,] matrix, bool isDerivative)
    {
        int numberOfRows = matrix.GetLength(0);
        int numberOfColumns = matrix.GetLength(1);
        double[,] result = new double[numberOfRows, numberOfColumns];

        for (int row = 0; row < numberOfRows; row++)
        {
            for (int column = 0; column < numberOfColumns; column++)
            {
                double sigmoidOutput =
                    result[row, column] = 1 / (1 + Math.Exp(-matrix[row, column]));

                double derivativeSigmoidOutput =
                    result[row, column] = matrix[row, column] * (1 - matrix[row, column]);

                result[row, column] = isDerivative
                    ? derivativeSigmoidOutput
                    : sigmoidOutput;
            }
        }

        return result;
    }

    public void Train(double[,] trainingInputs, double[,] trainingOutputs, int numberOfIterations)
    {
        for (int iteration = 0; iteration < numberOfIterations; iteration++)
        {
            double[,] output = Think(trainingInputs);
            double[,] error = PerformOperation(trainingOutputs, output, OPERATION.Subtract);
            double[,] adjustment =
                DotProduct(
                    Transpose(trainingInputs),
                    PerformOperation(error, Activate(output, true), OPERATION.Multiply)
                );

            _weights = PerformOperation(_weights, adjustment, OPERATION.Add);
        }
    }

    private double[,] DotProduct(double[,] matrix1, double[,] matrix2)
    {
        int numberOfRowsInMatrix1 = matrix1.GetLength(0);
        int numberOfColumnsInMatrix1 = matrix1.GetLength(1);

        int numberOfRowsInMatrix2 = matrix2.GetLength(0);
        int numberOfColumnsInMatrix2 = matrix2.GetLength(1);

        double[,] result =
            new double[numberOfColumnsInMatrix1, numberOfColumnsInMatrix2];

        for (int rowInMatrix1 = 0; rowInMatrix1 < numberOfRowsInMatrix1; rowInMatrix1++)
        {
            for (int columnInMatrix2 = 0; columnInMatrix2 < numberOfColumnsInMatrix2; columnInMatrix2++)
            {
                double sum = 0;

                for (int columnInMatrix1 = 0; columnInMatrix1 < numberOfColumnsInMatrix1; columnInMatrix1++)
                {
                    sum += matrix1[rowInMatrix1, columnInMatrix1] *
                           matrix2[columnInMatrix1, columnInMatrix2];
                }

                result[rowInMatrix1, columnInMatrix2] = sum;
            }
        }

        return result;
    }

    public double[,] Think(double[,] inputs)
    {
        return Activate(DotProduct(inputs, _weights), false);
    }

    private double[,] PerformOperation(double[,] matrix1, double[,] matrix2, OPERATION operation)
    {
        int rows = matrix1.GetLength(0);
        int columns = matrix1.GetLength(1);

        double[,] result = new double[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                switch (operation)
                {
                    case OPERATION.Multiply:
                        result[i, j] = matrix1[i, j] * matrix2[i, j];
                        break;

                    case OPERATION.Add:
                        result[i, j] = matrix1[i, j] + matrix2[i, j];
                        break;

                    case OPERATION.Subtract:
                        result[i, j] = matrix1[i, j] - matrix2[i, j];
                        break;
                }
            }
        }

        return result;
    }

    private double[,] Transpose(double[,] matrix)
        => matrix
            .Cast<double>()
            .ToArray()
            .Transpose(matrix.GetLength(0), matrix.GetLength(1));
}