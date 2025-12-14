// PageModels/RiskListPageModel.cs
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using Tectonic.Models;
using Tectonic.Services;

namespace Tectonic.PageModels
{
    public partial class RiskListPageModel : ObservableObject
    {
        private readonly EarthquakeApiService _apiService;

        [ObservableProperty]
        private bool _isBusy;

        // Ganti nama dari 'Projects' menjadi 'RiskyLocations'
        [ObservableProperty]
        private ObservableCollection<RiskyLocation> _riskyLocations;

        public RiskListPageModel(EarthquakeApiService apiService)
        {
            _apiService = apiService;
            RiskyLocations = new ObservableCollection<RiskyLocation>();
        }

        // Command ini akan dipanggil oleh Behavior 'Appearing' di XAML
        [RelayCommand]
        private async Task AppearingAsync()
        {
            // Panggil LoadLocationsCommand saat halaman muncul untuk pertama kali
            // jika koleksi masih kosong.
            if (RiskyLocations.Count == 0)
            {
                await LoadLocationsCommand.ExecuteAsync(null);
            }
        }

        // Command ini akan diikat ke AddButton/RefreshButton
        [RelayCommand]
        private async Task LoadLocationsAsync()
        {
            if (IsBusy)
                return;

            IsBusy = true;
            try
            {
                var locationsData = await _apiService.GetRiskyLocationsAsync();

                RiskyLocations.Clear();
                foreach (var loc in locationsData)
                {
                    RiskyLocations.Add(loc);
                }
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}