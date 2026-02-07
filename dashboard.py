import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import json

# Загрузка данных
@st.cache_data
def load_file():
    df = pd.read_csv('ecommerce_sales_data_upd.csv')
    df['Year'] = pd.to_datetime(df['order_date']).dt.year
    df['Month'] = pd.to_datetime(df['order_date']).dt.month
    return df

# Предварительный анализ
def pre_analystics(df):
    analytics = {
        'sales_by_year': df.groupby('Year')['sales'].sum().reset_index(),
        'sales_by_month': df.groupby(['Year', 'Month'])['sales'].sum().reset_index(),
        'profit_by_year': df.groupby('Year')['profit'].sum().reset_index(),
        'profit_by_month': df.groupby('Month')['profit'].sum().reset_index(),
        'top_region': df.groupby('region')['sales'].sum().nlargest(5).reset_index(),
        'top_product': df.groupby('product_name')['sales'].sum().nlargest(5).reset_index(),
        'top_categories': df.groupby('category')['sales'].sum().reset_index(),
        'top_quantity': df.groupby('category')['quantity'].count().reset_index(),
        'profit_by_product': df.groupby('product_name')['profit'].sum().nlargest(5).reset_index(),
        'profit_by_category': df.groupby('category')['profit'].sum().reset_index(),
        'avg_check_by_region': df.groupby('region')['sales'].mean().reset_index(),
        'avg_check_by_category': df.groupby('category')['sales'].mean().reset_index(),
        'sales_by_region_month': df.groupby(['region', 'Month'])['sales'].sum().reset_index()
    }
    return analytics

def ask_llm(question, df_sample):
    columns_description = {
        'order_date': 'Дата заказа',
        'product_name': 'Название товара',
        'category': 'Категория товара',
        'region': 'Регион продажи',
        'quantity': 'Количество проданных единиц',
        'sales': 'Выручка от продажи',
        'profit': 'Прибыль от продажи'
    }

    prompt = f"""
    Ответь на русском языке.

    Структура данных: {columns_description}

    Примеры значений:
    {df_sample.head(3).to_dict('records')}

    Доступные задачи:
    - group_by_month: сгруппировать данные по месяцам и рассчитать сумму выручки или прибыли.
    - group_by_region: сгруппировать данные по регионам и рассчитать сумму выручки или прибыли.
    - group_by_category: сгруппировать данные по категориям и рассчитать сумму выручки или прибыли.
    - find_max_profit: найти месяц, регион или категорию с максимальной прибылью.
    - find_max_sales: найти месяц, регион или категорию с максимальной выручкой.

    Пользователь спрашивает: "{question}".

    Сгенерируй ответ в формате JSON с ключами:
    - "task": одна из доступных задач.
    - "metric": метрика для расчёта (например, "profit" или "sales").
    - "group_by": поле для группировки (например, "Month", "region", "category").
    - "explanation": объяснение для пользователя на русском языке.
    """

    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

def process_llm_response(llm_response, df):
    try:
        llm_response = llm_response.strip()
        if not llm_response.startswith('{'):
            llm_response = llm_response.split('{', 1)[1].rsplit('}', 1)[0]
            llm_response = '{' + llm_response + '}'

        instruction = json.loads(llm_response)
        if 'error' in instruction:
            return instruction['error']

        task = instruction.get('task')
        metric = instruction.get('metric', 'sales')
        group_by = instruction.get('group_by')

        if task in ['group_by_month', 'group_by_region', 'group_by_category']:
            result = df.groupby(group_by)[metric].sum().reset_index()
            return {'data': result, 'x': group_by, 'y': metric, 'task': task}
        elif task in ['find_max_profit', 'find_max_sales']:
            metric = 'profit' if task == 'find_max_profit' else 'sales'
            grouped = df.groupby(group_by)[metric].sum().reset_index()
            max_row = grouped.loc[grouped[metric].idxmax()]
            return {'data': max_row.to_frame().T, 'x': group_by, 'y': metric, 'task': task}
        else:
            return "Не удалось обработать запрос."
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Визуализация
def plot_sales_for_year(data):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='Year', y='sales', ax=ax)
    ax.set_xticks(data['Year'])
    ax.set_xticklabels(data['Year'].astype(str))
    ax.set_title('Динамика продаж по годам')
    ax.set_xlabel('Год')
    ax.set_ylabel('Выручка, тыс.')
    ax.set_ylabel('Регионы')
    return fig

def plot_top_products(data):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='sales', y='product_name', ax=ax)
    ax.set_title('Топ-5 продуктов по выручке')
    ax.set_xlabel('Выручка, тыс. $')
    ax.set_ylabel('Название продукта')
    return fig

def plot_profit_by_category(data):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='profit', y='category', ax=ax)
    ax.set_title('Прибыль по категориям')
    ax.set_xlabel('Прибыль, тыс. $')
    ax.set_ylabel('Название категории')
    return fig

def plot_top_region(data):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='sales', y='region', ax=ax)
    ax.set_title('Топ регионов по выручке')
    ax.set_xlabel('Выручка, тыс. $')
    ax.set_ylabel('Регионы')
    return fig

def plot_avg_category(data):
    fig, ax = plt.subplots()
    ax.pie(data['sales'], labels=data['category'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Средний чек выручки по категориям')
    return fig

def plot_sales_by_month(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for year, group in data.groupby('Year'):
        ax.plot(group['Month'], group['sales'], label=f'Год {year}')
    ax.set_title('Продажи по месяцам')
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Выручка, тыс.$')
    ax.set_xticks(range(1, 13))
    ax.legend()
    return fig

def main():
    st.title('Дашборд продаж')
    df = load_file()
    analytics = pre_analystics(df)

    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input('Задайте вопрос о данных:')

    with col2:
        st.sidebar.title('Навигация')

    plot_option = st.sidebar.selectbox(
        'Выберите график:',
        ['Динамика продаж по годам', 'Топ-5 продуктов по выручке', 'Прибыль по категориям', 'Продажи по месяцам', 'Топ регионов по выручке', 'Средний чек выручки по категориям']
    )

    if user_question:
        llm_response = ask_llm(user_question, df)
        result = process_llm_response(llm_response, df)

        if isinstance(result, dict) and 'data' in result:
            data = result['data']

            fig, ax = plt.subplots(figsize=(10, 6))
            if result['task'] in ['find_max_profit', 'find_max_sales'] and isinstance(data, pd.DataFrame):
                sns.barplot(data=data, x=result['x'], y=result['y'], ax=ax)
                ax.set_title(f"{data[result['y']].max()} (максимальное значение)")
            elif result['task'] == 'top_products':
                sns.barplot(data=data, x=result['y'], y=result['x'], ax=ax)
                ax.set_title('Топ товары по выручке')
            else:
                if result['x'] == 'Month':
                    sns.lineplot(data=data, x=result['x'], y=result['y'], ax=ax)
                else:
                    sns.barplot(data=data, x=result['x'], y=result['y'], ax=ax)
                ax.set_title(f"{result['y'].capitalize()} по {result['x']}")
            st.pyplot(fig)
            st.dataframe(data)
        elif isinstance(result, str):
            st.write(result)
    else:
        if plot_option == 'Динамика продаж по годам':
            st.pyplot(plot_sales_for_year(analytics['sales_by_year']))
        elif plot_option == 'Топ-5 продуктов по выручке':
            st.pyplot(plot_top_products(analytics['top_product']))
        elif plot_option == 'Прибыль по категориям':
            st.pyplot(plot_profit_by_category(analytics['profit_by_category']))
        elif plot_option == 'Продажи по месяцам':
            st.pyplot(plot_sales_by_month(analytics['sales_by_month']))
        elif plot_option == 'Топ регионов по выручке':
            st.pyplot(plot_top_region(analytics['top_region']))
        elif plot_option == 'Средний чек выручки по категориям':
            st.pyplot(plot_avg_category(analytics['avg_check_by_category']))

    if st.checkbox('Показать сырые данные'):
        st.dataframe(df.head())

if __name__ == '__main__':
    main()
