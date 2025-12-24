using TorchSharp;

int[,] maze1 =
{
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0 }, //row 0
    { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 }, //row 1
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 }, //row 2
    { 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0 }, //row 3
    { 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0 }, //row 4
    { 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 }, //row 5
    { 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 }, //row 6
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 }, //row 7
    { 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0 }, //row 8
    { 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 }, //row 9
    { 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0 }, //row 10
    { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 } //row 11 (start position is (11, 5))
};
const string up = "UP";
const string down = "DOWN";
const string left = "LEFT";
const string right = "RIGHT";
string[] actions = [up, down, left, right];

int[,] rewards;
const int wallRewardValue = -500;
const int floorRewardValue = -10;
const int goalRewardValue = 500;

torch.Tensor qValues;

const float epsilon = 0.95f; //exploration factor
const float discountFactor = 0.8f; //discount factor
const float learningRate = 0.9f; //learning rate
const int episodes = 1500; //number of training episodes    
const int startRow = 11;
const int startCol = 5;
SetUpRewards(maze1, wallRewardValue, floorRewardValue, goalRewardValue);
SetupQValues(maze1);
TrainModel(maze1, floorRewardValue, epsilon, discountFactor, learningRate, episodes);
NavigateMaze(maze1, startRow, startCol, floorRewardValue, wallRewardValue);
return;

void SetupQValues(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeCols = maze.GetLength(1);
    qValues = torch.zeros(mazeRows, mazeCols, 4);
}

bool HasHitWallOrEndOfMaze(int currentRow, int currentCol, int floorValue)
{
    return rewards[currentRow, currentCol] != floorValue;
}

long DetermineNextAction(int currentRow, int currentCol, double epsilon)
{
    var rand = new Random();
    var randomBetween0And1 = rand.NextDouble();
    var nextAction = randomBetween0And1 < epsilon
        ? torch.argmax(qValues[currentRow, currentCol]).item<long>()
        : rand.Next(0, 4);
    return nextAction;
}

(int, int) MoveOnSpace(int[,] maze, int currentRow, int currentCol, long currentAction)
{
    int mazeRows = maze.GetLength(0);
    int mazeCols = maze.GetLength(1);
    int newRow = currentRow;
    int newCol = currentCol;
    if (actions[currentAction] == up && currentRow > 0)
    {
        newRow--;
    }
    else if (actions[currentAction] == down && currentRow < mazeRows - 1)
    {
        newRow++;
    }
    else if (actions[currentAction] == left && currentCol > 0)
    {
        newCol--;
    }
    else if (actions[currentAction] == right && currentCol < mazeCols - 1)
    {
        newCol++;
    }

    return (newRow, newCol);
}

void TrainModel(int[,] maze, int floorValue, double epsilon, double discountFactor, double learningRate, int episodes)
{
    for (int episode = 0; episode < episodes; episode++)
    {
        Console.WriteLine($"------Starting episode {episode}------");
        int currentRow = 11;
        int currentCol = 5;
        while (!HasHitWallOrEndOfMaze(currentRow, currentCol, floorValue))
        {
            long currentAction = DetermineNextAction(currentRow, currentCol, epsilon);
            var priviousRow = currentRow;
            var priviousCol = currentCol;
            var nextMove = MoveOnSpace(maze, currentRow, currentCol, currentAction);
            currentRow = nextMove.Item1;
            currentCol = nextMove.Item2;
            var reward = rewards[currentRow, currentCol];
            var priviousQValue = qValues[priviousRow, priviousCol, currentAction].item<float>();
            var temporalDifference = reward +
                discountFactor * torch.max(qValues[currentRow, currentCol]).item<float>() - priviousQValue;
            var newQValue = priviousQValue + (learningRate * temporalDifference);
            qValues[priviousRow, priviousCol, currentAction] = newQValue;
        }

        Console.WriteLine($"------Ending episode {episode}------");
    }

    Console.WriteLine("Training complete"
    );
}

List<int[]> NavigateMaze(int[,] maze, int startRow, int startCol, int floorValue, int wallValue)
{
    List<int[]> path = new List<int[]>();
    if (HasHitWallOrEndOfMaze(startRow, startCol, floorValue))
        return [];
    else
    {
        int currentRow = startRow;
        int currentCol = startCol;
        path = [[currentRow, currentCol]];
        while (!HasHitWallOrEndOfMaze(currentRow, currentCol, floorValue))
        {
            var nextAction = (int)DetermineNextAction(currentRow, currentCol, 1.0f);
            var nextMove = MoveOnSpace(maze, currentRow, currentCol, nextAction);
            currentRow = nextMove.Item1;
            currentCol = nextMove.Item2;
            if (rewards[currentRow, currentCol] != wallValue)
                path.Add([currentRow, currentCol]);
            else
            {
                continue;
            }
        }
    }

    int moveCount = 1;
    for (int i = 0; i < path.Count; i++)
    {
        Console.WriteLine("Move" + moveCount + ": (");
        foreach (var element in path[i])
            Console.WriteLine(" " + element + ", ");
        Console.WriteLine(")");
        Console.WriteLine();
        moveCount++;
    }

    return path;
}

void SetUpRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    int mazeRows = maze.GetLength(0);
    int mazeCols = maze.GetLength(1);
    rewards = new int[mazeRows, mazeCols];
    for (int row = 0; row < mazeRows; row++)
    {
        for (int col = 0; col < mazeCols; col++)
        {
            switch (maze[row, col])
            {
                case 0:
                    rewards[row, col] = wallValue;
                    break;
                case 1:
                    rewards[row, col] = floorValue;
                    break;
                case 2:
                    rewards[row, col] = goalValue;
                    break;
            }
        }
    }
}