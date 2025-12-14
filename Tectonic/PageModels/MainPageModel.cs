// PageModels/MainPageModel.cs
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Tectonic.Models;
using Tectonic.Services;

namespace Tectonic.PageModels
{
    public partial class MainPageModel : ObservableObject
    {
        private readonly EarthquakeApiService _apiService;

        // Properti untuk Status Loading
        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(IsNotBusy))]
        private bool _isBusy;
        public bool IsNotBusy => !_isBusy;

        // Data untuk Chart
        [ObservableProperty]
        private ObservableCollection<SeveritySummary> _severitySummary;

        // Data untuk daftar "Lokasi Paling Rawan" (sebelumnya Projects)
        [ObservableProperty]
        private ObservableCollection<RiskyLocation> _topRiskyLocations;

        // Data untuk "Ringkasan Hasil Terbaik" (sebelumnya Tasks)
        [ObservableProperty]
        private BestCombination _gaResult;

        // Properti untuk Peta Risiko
        [ObservableProperty]
        private ImageSource _riskMapSource;

        public MainPageModel(EarthquakeApiService apiService)
        {
            _apiService = apiService;
            SeveritySummary = new ObservableCollection<SeveritySummary>();
            TopRiskyLocations = new ObservableCollection<RiskyLocation>();
        }

        [RelayCommand]
        async Task AppearingAsync()
        {
            // Panggil refresh data saat halaman pertama kali muncul
            await RefreshDataAsync();
        }

        [RelayCommand]
        async Task RefreshDataAsync()
        {
            if (IsBusy) return;

            IsBusy = true;
            try
            {
                // Ambil semua data dari API secara bersamaan
                var riskyLocationsTask = _apiService.GetRiskyLocationsAsync();
                var bestCombinationTask = _api_service.GetBestCombinationAsync();

                await Task.WhenAll(riskyLocationsTask, bestCombinationTask);

                var allRiskyLocations = await riskyLocationsTask;
                GaResult = await bestCombinationTask;

                // 1. Proses data untuk Chart
                var summary = allRiskyLocations
                    .GroupBy(l => l.LabelBahaya)
                    .Select(g => new SeveritySummary { LabelBahaya = g.Key, Jumlah = g.Count() })
                    .OrderBy(s => s.LabelBahaya);

                SeveritySummary.Clear();
                foreach (var item in summary)
                {
                    SeveritySummary.Add(item);
                }

                // 2. Proses data untuk "Top Lokasi Rawan"
                var topLocations = allRiskyLocations
                    .OrderByDescending(l => l.Magnitudo)
                    .Take(5); // Ambil 5 teratas

                TopRiskyLocations.Clear();
                foreach (var loc in topLocations)
                {
                    TopRiskyLocations.Add(loc);
                }

                // 3. Muat gambar Peta
                RiskMapSource = ImageSource.FromUri(new Uri(_apiService.GetRiskMapUrl()));
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}