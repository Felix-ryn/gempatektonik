// PageModels/ResultSummaryPageModel.cs
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Threading.Tasks;
using Tectonic.Models;
using Tectonic.Services;

namespace Tectonic.PageModels
{
    // Kita gunakan [INotifyPropertyChanged] jika tidak ada command
    // atau gunakan [ObservableObject] jika ada command, keduanya sama-sama dari CommunityToolkit.Mvvm
    [INotifyPropertyChanged]
    public partial class ResultSummaryPageModel
    {
        private readonly EarthquakeApiService _apiService;

        // Properti untuk menyimpan hasil dari Genetic Algorithm
        [ObservableProperty]
        private BestCombination _gaResult;

        // Properti untuk menyimpan sumber gambar Peta Risiko
        [ObservableProperty]
        private ImageSource _riskMapSource;

        [ObservableProperty]
        private bool _isLoading = true;

        public ResultSummaryPageModel(EarthquakeApiService apiService)
        {
            _apiService = apiService;
        }

        // Metode ini akan kita panggil dari code-behind saat halaman muncul
        public async Task LoadDataAsync()
        {
            if (!IsLoading) IsLoading = true;

            try
            {
                // Ambil data dari API secara bersamaan untuk efisiensi
                var gaResultTask = _apiService.GetBestCombinationAsync();
                var mapUrl = _apiService.GetRiskMapUrl();

                GaResult = await gaResultTask;
                RiskMapSource = ImageSource.FromUri(new Uri(mapUrl));
            }
            catch (System.Exception ex)
            {
                // Handle error (misal: tampilkan pesan)
                System.Diagnostics.Debug.WriteLine($"Error loading summary data: {ex.Message}");
            }
            finally
            {
                IsLoading = false;
            }
        }
    }
}