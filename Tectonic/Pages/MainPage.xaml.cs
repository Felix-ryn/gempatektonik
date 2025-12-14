using System;
using Tectonic.Models;
using Tectonic.PageModels;

namespace Tectonic.Pages
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();
            BindingContext = model;
        }

        private async void OnRiskyLocationsClicked(object sender, EventArgs e)
        {
            await Shell.Current.GoToAsync(nameof(RiskListPage));
        }

        private async void OnResultsClicked(object sender, EventArgs e)
        {
            await Shell.Current.GoToAsync(nameof(ResultSummaryPage));
        }

    }
}