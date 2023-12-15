from django.shortcuts import render, redirect
from .stock_model import model_func, get_stock_data, fetch_news, aggregate_sentiments,combine_data, add_indicators

def stock_view(request):
    if request.method == "POST":
        ticker = request.POST.get("ticker")
        start_date = request.POST.get("start_date")

        # Fetch and process data
        # Date must be str of form YYYY-MM-DD
        stock_data = get_stock_data(ticker,start_date)
        stock_data = add_indicators(stock_data)
        news_data = fetch_news(ticker)
        sentiment_data = aggregate_sentiments(news_data)
        combined_data = combine_data(stock_data, sentiment_data)

        # Call your model function
        processed_data = model_func(combined_data)

        #Store data in session
        request.session['processed_data'] = processed_data

        #Direct to loading page
        return redirect('loading-view')

    return render(request, "stockapphtml/index.html")

def loading_view(request):
    #Loading page while data is being process through the model
    return render(request, "stockapphtml/loading.html")

def result_view(request):
    #Retrieve processed data
    data = request.session.get('processed_data', None)
    return render(request, "stockapphtml/result.html", {"data": data})