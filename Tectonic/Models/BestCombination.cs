// Models/BestCombination.cs
using Newtonsoft.Json;

namespace Tectonic.Models
{
    public class BestCombination
    {
        [JsonProperty("Magnitudo")]
        public double Magnitudo { get; set; }

        [JsonProperty("Kedalaman (km)")]
        public double KedalamanKm { get; set; }

        [JsonProperty("Area Terdampak (km)")]
        public double AreaTerdampakKm { get; set; }

        [JsonProperty("FitnessScore")]
        public double FitnessScore { get; set; }
    }
}