// Services/EarthquakeApiService.cs
using Newtonsoft.Json;
using System.Net.Http;
using System.Threading.Tasks;
using System.Collections.Generic;
using Tectonic.Models;

namespace Tectonic.Services
{
    public class EarthquakeApiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public EarthquakeApiService()
        {
            _httpClient = new HttpClient();

            // PENTING: Sesuaikan alamat IP ini!
            // Gunakan 10.0.2.2 untuk Android Emulator jika API jalan di PC yang sama
            // Gunakan alamat IP lokal Anda (misal: 192.168.1.5) jika menjalankan di HP fisik
            // Gunakan localhost jika menjalankan di Windows
            _baseUrl = DeviceInfo.Platform == DevicePlatform.Android
                        ? "http://10.0.2.2:5000"
                        : "http://localhost:5000";
        }

        public async Task<List<RiskyLocation>> GetRiskyLocationsAsync()
        {
            var url = $"{_baseUrl}/api/risky-locations";
            var response = await _httpClient.GetAsync(url);
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                return JsonConvert.DeserializeObject<List<RiskyLocation>>(content);
            }
            return new List<RiskyLocation>(); // Kembalikan list kosong jika gagal
        }

        public async Task<BestCombination> GetBestCombinationAsync()
        {
            var url = $"{_baseUrl}/api/best-combination";
            var response = await _httpClient.GetAsync(url);
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                return JsonConvert.DeserializeObject<BestCombination>(content);
            }
            return null; // Kembalikan null jika gagal
        }

        // Helper untuk mendapatkan URL peta langsung
        public string GetRiskMapUrl()
        {
            return $"{_baseUrl}/api/risk-map";
        }
    }
}