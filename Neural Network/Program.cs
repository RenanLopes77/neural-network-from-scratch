using System.Text.Json;

var layers = new[] { 2, 2, 2, 1 };
var neuralNetwork = new NeuralNetwork(layers);
neuralNetwork.ConnectNeurons();

Console.Write(JsonSerializer.Serialize(neuralNetwork, new JsonSerializerOptions { WriteIndented = true }));

public class NeuralNetwork
{
    public List<Layer> Layers { get; private set; }
    public double Prediction { get; private set; }
    public ILossFuntion Loss { get; private set; }

    public NeuralNetwork(int[] neuronsPerLayer)
    {
        Layers = [];
        for (int index = 0; index < neuronsPerLayer.Length; index++)
        {
            var lastIndex = neuronsPerLayer.Length - 1;
            LayerType layerType = index switch
            {
                0 => LayerType.Input,
                var _ when index == lastIndex => LayerType.Output,
                _ => LayerType.None
            };
            int neuronsNextLayer = index < lastIndex ? neuronsPerLayer[index + 1] : 0;

            var layer = new Layer(layerType, neuronsPerLayer[index], neuronsNextLayer);
            Layers.Add(layer);
        }
    }

    public void ConnectNeurons()
    {
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            var neurons = Layers[i].Neurons;
            for (int ii = 0; ii < neurons.Count; ii++)
            {
                var connections = neurons[ii].Connections;
                for (int iii = 0; iii < connections.Count; iii++)
                {
                    var connection = connections[iii];
                    var neuron = Layers[i + 1].Neurons[iii];
                    Console.WriteLine($"Neuron {i}: " + neurons[ii].ID + $" => Neuron {i + 1}: " + neuron.ID + " Connection: " + iii);
                    connection.SetNeuron(neuron);
                }
            }
        }
    }
}

public enum LayerType
{
    None,
    Input,
    Output
}
public class Layer
{
    public List<Neuron> Neurons { get; private set; }
    public LayerType Type { get; private set; }

    public Layer(LayerType type, int neurons, int neuronsNextLayer)
    {
        Type = type;
        Neurons = [];
        IActivationFunction activation = type switch
        {
            LayerType.Input => new NoActivation(),
            LayerType.Output => new LogisticSigmoid(),
            _ => new Tanh()
        };
        for (int i = 0; i < neurons; i++)
        {
            Neurons.Add(new Neuron(activation, neuronsNextLayer));
        }
    }
}

public class Neuron
{
    public Guid ID { get; private set; }
    public double Input { get; private set; }
    public double Output { get; private set; }
    public double Bias { get; private set; }
    public List<Connection> Connections { get; private set; }
    public IActivationFunction Activation { get; private set; }

    public Neuron(IActivationFunction activation, int neuronsNextLayer)
    {
        ID = Guid.NewGuid();
        Console.WriteLine("Neuron: " + ID);
        Bias = 0;
        Activation = activation;
        Connections = [];
        for (int i = 0; i < neuronsNextLayer; i++)
        {
            Connections.Add(new Connection());
        }
    }
}

public class Connection
{
    public double Weight { get; private set; }
    public Neuron NeuronTarget { get; private set; }

    public Connection()
    {
        var random = new Random();
        Weight = random.NextDouble();
    }

    public void SetNeuron(Neuron neuron)
    {
        NeuronTarget = neuron;
    }
}

public interface ILossFuntion
{
}

public interface IActivationFunction
{
    public double Activate(double x);
}

public class LogisticSigmoid : IActivationFunction
{
    public double Activate(double x)
    {
        return 1 / (1 + Math.Exp(-x)); // Compresses the input it receives into output between 0 and 1 (x = 0.5 => output is 1)
    }
}

public class Tanh : IActivationFunction
{
    public double Activate(double x)
    {
        return Math.Tanh(x); // Compresses the input it receives into output between -1 and 1 ... used in our hidden layers
    }
}

public class NoActivation : IActivationFunction
{
    public double Activate(double x)
    {
        return x;
    }
}