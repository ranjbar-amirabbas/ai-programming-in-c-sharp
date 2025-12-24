namespace _02_NueralNetwork;

public static class Extensions
{
    public static double[,] Transpose(this double[] array, int rows, int columns)
    {
        double[,] result = new double[columns, rows];

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                result[column, row] = array[row * columns + column];
            }
        }

        return result;
    }
}