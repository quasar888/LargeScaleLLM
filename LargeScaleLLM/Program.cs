using System;
using System.Threading;
using System.Threading.Tasks;

namespace LargeScaleLLM
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("Training Neural Network...");

            NeuralNetwork nn = new NeuralNetwork(5, 100); // 5 Neurons, 100 Features

            Thread trainingThread = new Thread(() =>
            {
                nn.Train(epochs: 1000, batchSize: 500);
            });

            trainingThread.Start();
            trainingThread.Join(); // Wait for the training to complete before exiting
        }
    }

    class NeuralNetwork
    {
        private int _inputSize, _hiddenSize;
        private double learningRate = 0.00001;
        private double[][] _weights;
        private double[][] _errors;

        public NeuralNetwork(int inputSize, int hiddenSize)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            InitializeWeights();
            _errors = new double[_hiddenSize][]; // ✅ Initialize `_errors`
            for (int i = 0; i < _hiddenSize; i++)
            {
                _errors[i] = new double[_inputSize]; // ✅ Initialize each row
            }
        }

        private void InitializeWeights()
        {
            Random rand = new Random();
            _weights = new double[_hiddenSize][];
            for (int i = 0; i < _hiddenSize; i++)
            {
                _weights[i] = new double[_inputSize];
                for (int j = 0; j < _inputSize; j++)
                {
                    _weights[i][j] = Math.Round(rand.NextDouble(), 3) + 0.01;
                }
            }
        }

        private double[] ForwardPass(double[] input)
        {
            double[] output = new double[_hiddenSize];
            for (int i = 0; i < _hiddenSize; i++)
            {
                output[i] = 0;
                for (int j = 0; j < _inputSize; j++)
                {
                    output[i] += input[j] * _weights[i][j];
                }
                output[i] = Sigmoid(output[i]);
            }
            return output;
        }

        private void BackwardPass(double[] input, double[] expectedOutput, double[] actualOutput)
        {
            for (int i = 0; i < _hiddenSize; i++)
            {
                double error = expectedOutput[i] - actualOutput[i];
                double delta = error * actualOutput[i] * (1 - actualOutput[i]); // Derivative of sigmoid

                for (int j = 0; j < _inputSize; j++)
                {
                    _errors[i][j] = delta * input[j]; // ✅ No longer causes NullReferenceException
                    _weights[i][j] += learningRate * _errors[i][j]; // Weight Update
                }
            }
        }

        public void Train(int epochs, int batchSize)
        {
            Random rand = new Random();
            long totalError = 0; // ✅ Use `long` for thread-safe updates

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[][] inputBatch = DataHandler.GenerateRandomData(batchSize, _inputSize);
                double[][] targetBatch = DataHandler.GenerateTargetData(batchSize, _hiddenSize);

                totalError = 0; // Reset error for each epoch

                Parallel.For(0, batchSize, i =>
                {
                    double[] input = inputBatch[i];
                    double[] expectedOutput = targetBatch[i];

                    double[] actualOutput = ForwardPass(input);
                    BackwardPass(input, expectedOutput, actualOutput);

                    // ✅ Use Interlocked.Add to prevent race conditions
                    long errorSum = 0;
                    for (int j = 0; j < _hiddenSize; j++)
                    {
                        errorSum += (long)(Math.Abs(expectedOutput[j] - actualOutput[j]) * 1000); // Scale to avoid precision loss
                    }
                    Interlocked.Add(ref totalError, errorSum);
                });

                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch} - Total Error: {totalError / (batchSize * 1000.0)}");
                }
            }
            Console.WriteLine("Training Complete!");
        }

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    }

    class DataHandler
    {
        public static double[][] GenerateRandomData(int samples, int features)
        {
            Random rand = new Random();
            double[][] data = new double[samples][];
            for (int i = 0; i < samples; i++)
            {
                data[i] = new double[features];
                for (int j = 0; j < features; j++)
                {
                    data[i][j] = rand.NextDouble();
                    //Console.WriteLine("data[i][j]" + i+" " +j+" " + data[i][j]);
                }
            }
            return data;
        }

        public static double[][] GenerateTargetData(int samples, int features)
        {
            double[][] targets = new double[samples][];
            for (int i = 0; i < samples; i++)
            {
                targets[i] = new double[features];
                for (int j = 0; j < features; j++)
                {
                    targets[i][j] = 0.5; // Dummy target values
                }
            }
            return targets;
        }
    }
}
