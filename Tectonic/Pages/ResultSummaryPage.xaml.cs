// Pages/ResultSummaryPage.xaml.cs
using Tectonic.PageModels;

namespace Tectonic.Pages;

public partial class ResultSummaryPage : ContentPage
{
    private readonly ResultSummaryPageModel _viewModel;

    public ResultSummaryPage(ResultSummaryPageModel vm)
    {
        InitializeComponent();
        _viewModel = vm;
        BindingContext = _viewModel;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        // Panggil metode untuk memuat data dari ViewModel
        await _viewModel.LoadDataAsync();
    }
}