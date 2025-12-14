// Models/RiskyLocation.cs
using Newtonsoft.Json;

namespace Tectonic.Models
{
    public class RiskyLocation
    {
        [JsonProperty("Tanggal")]
        public string Tanggal { get; set; }

        [JsonProperty("Lintang")]
        public double Lintang { get; set; }

        [JsonProperty("Bujur")]
        public double Bujur { get; set; }

        [JsonProperty("Magnitudo")]
        public double Magnitudo { get; set; }

        [JsonProperty("Kedalaman (km)")]
        public double KedalamanKm { get; set; }

        [JsonProperty("Lokasi")]
        public string Lokasi { get; set; }

        [JsonProperty("Area Terdampak (km)")]
        public double AreaTerdampakKm { get; set; }

        [JsonProperty("LabelBahaya")]
        public string LabelBahaya { get; set; }
    }
}