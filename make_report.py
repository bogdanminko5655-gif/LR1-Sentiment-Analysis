"""Формує .docx звіт за реальними метриками з results/metrics.json.

python main.py
python make_report.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Lab1_Report.docx'
DIR=ROOT/'results'


def style_report(doc):
    sec=doc.sections[0]
    sec.top_margin=Cm(2.0)
    sec.bottom_margin=Cm(2.0)
    sec.left_margin=Cm(2.5)
    sec.right_margin=Cm(2.0)
    sec.header_distance=Cm(1)
    normal=doc.styles['Normal']
    normal.font.name='Times New Roman'
    normal.font.size=Pt(11.5)
    normal.paragraph_format.space_after=Pt(4)
    normal.paragraph_format.line_spacing=1.17
    for st_name, size, col in [('Title',20,RGBColor(25,49,92)),
                               ('Heading 1',15,RGBColor(25,49,92)),
                               ('Heading 2',13,RGBColor(34,64,112))]:
        st=doc.styles[st_name]
        st.font.name='Times New Roman'; st.font.size=Pt(size)
        st.font.bold=True; st.font.color.rgb=col
    st=doc.styles.add_style('Small note', WD_STYLE_TYPE.PARAGRAPH)
    st.font.name='Times New Roman'; st.font.size=Pt(10)
    st.font.color.rgb=RGBColor(86,91,99)
    sec.header.paragraphs[0].text='КДПУ  •  Лабораторна робота № 1  •  Машинне навчання'
    sec.header.paragraphs[0].style='Small note'
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.text='2026  •  Навчальний проєкт'
    footer.style='Small note'


def p(doc, text, style=None):
    return doc.add_paragraph(text, style=style)


def table(doc, headers, rows):
    tb=doc.add_table(rows=1, cols=len(headers), style='Light Shading Accent 1')
    tb.alignment=WD_TABLE_ALIGNMENT.CENTER
    for cell, label in zip(tb.rows[0].cells, headers): cell.text=str(label)
    for line in rows:
        cells=tb.add_row().cells
        for cell, value in zip(cells, line): cell.text=str(value)
    doc.add_paragraph()
    return tb


def figure(doc, filename, caption):
    path=DIR/filename
    if path.exists():
        doc.add_picture(str(path), width=Cm(14.5))
        paragraph=doc.add_paragraph(caption, style='Small note')
        paragraph.alignment=WD_ALIGN_PARAGRAPH.CENTER


def build_report(metrics=None):
    doc=Document()
    style_report(doc)
    header=doc.add_paragraph()
    header.alignment=WD_ALIGN_PARAGRAPH.CENTER
    header.add_run('МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ\n').bold=True
    header.add_run('Криворізький державний педагогічний університет\n')
    p(doc, '')
    doc.add_paragraph('ЗВІТ\nз лабораторної роботи № 1', style='Title').alignment=WD_ALIGN_PARAGRAPH.CENTER
    t=doc.add_paragraph('«Застосування логістичної регресії для аналізу тональності текстів»')
    t.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p(doc, '')
    p(doc, '')
    p(doc, 'Виконав: студент групи І-23\nМінько Богдан')
    p(doc, 'Навчальна дисципліна: ___________________________')
    p(doc, 'Перевірив(ла): _________________________________')
    p(doc, '')
    p(doc, 'Кривий Ріг — 2026').alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()
    doc.add_heading('1. Мета та постановка задачі', level=1)
    p(doc, 'Мета — навчитися використовувати логістичну регресію для аналізу тональності текстів, реалізувати градієнтний спуск, оцінити якість класифікації на відкладеній вибірці та дослідити вплив гіперпараметрів.')
    p(doc, 'Потрібно підготувати корпус текстів, виконати токенізацію, видалення стоп-слів і стемінг, побудувати частотний словник для позитивного та негативного класів, навчити модель і дослідити її точність.')
    doc.add_heading('2. Вхідні дані', level=1)
    p(doc, 'Номер варіанта в наданих умовах не вказано. Як демонстраційний вибір використано корпус відгуків IMDb з набору Sentiment Labelled Sentences (UCI), який прямо наведений у методичних рекомендаціях. Корпус англомовний; мітка 0 означає негативну тональність, 1 — позитивну.')
    p(doc, 'Офіційний опис UCI: 1000 речень IMDb, по 500 для кожного класу. Дані походять із дослідження Kotzias та співавторів (2015), ліцензія CC BY 4.0.')
    if metrics:
        p(doc, f'Фактично прочитано {metrics["total"]} речень. Початковий поділ: {metrics["train"]} навчальних і {metrics["test"]} тестових (seed={metrics["seed"]}).')
    doc.add_heading('3. Попередня обробка текстів', level=1)
    p(doc, 'Програмна функція process_text виконує: переведення у нижній регістр; усунення URL; розбиття на англійські слова регулярним виразом; вилучення англійських стоп-слів; стемінг за алгоритмом Портера. Слова not, no, nor, never залишено, щоб не втратити заперечення.')
    p(doc, 'Приклад: «I did not enjoy this boring movie» → набір нормалізованих стемів. Стемінг зменшує кількість варіантів словоформ, а видалення службових слів скорочує зайвий шум.')
    doc.add_heading('4. Частотний словник та ознаки', level=1)
    p(doc, 'Для кожного стема та класу обчислюється число входжень у навчальному корпусі: freqs[(word, class)]. На його основі кожне речення описують три значення: x₀ = 1 (зсув), x₁ = сума позитивних частот його слів, x₂ = сума негативних частот.')
    p(doc, 'Для числової стійкості до x₁ і x₂ застосовано log(1+x) та стандартизацію за статистиками лише тренувальної вибірки. Це додаткова технічна нормалізація; вихідні три ознаки відповідають прикладу методички. Частотний словник не містить текстів валідації та тесту.')
    doc.add_heading('5. Реалізація логістичної регресії', level=1)
    p(doc, 'Ймовірність позитивного класу: σ(z) = 1 / (1 + exp(−z)), де z = Xθ. За порога 0,5 отримуємо клас 1, інакше — клас 0. Функція втрат — середня бінарна кросентропія (log loss).')
    p(doc, 'Градієнтне оновлення: θ := θ − α·Xᵀ(σ(Xθ)−y)/m. Обчислення реалізовано власною функцією gradient_descent у main.py (використано NumPy, без готового sklearn.LogisticRegression).')
    doc.add_heading('6. План експериментів', level=1)
    p(doc, 'Загальний поділ — 80% для розробки та 20% відкладеного тесту. Всередині 80% створено ще навчальну й валідаційну частини у співвідношенні 75:25 (60% / 20% / 20% від усього набору). Для кожного експерименту частотний словник обчислюється заново лише за його навчальними прикладами.')
    table(doc, ['Гіперпараметр','Досліджувані значення'],[
        ['Частка від доступних для підбору 60%','40%, 70%, 100%'],
        ['Швидкість навчання α','0,01; 0,05; 0,10'],
        ['Кількість ітерацій','100; 400; 1200']])
    p(doc, 'Загалом: 3 × 3 × 3 = 27 експериментів. Параметри вибираються за валідаційною точністю; якщо вона однакова — за нижчою log loss. Після вибору модель навчається повторно на всіх 80% початкових даних, а тест застосовується тільки для остаточної оцінки.')
    doc.add_heading('7. Результати та оцінка', level=1)
    if metrics:
        best=metrics['best_hyperparameters']
        table(doc,['Показник','Отримане значення'],[
            ['Точність (accuracy)', f'{metrics["accuracy"]*100:.2f}%'],
            ['Precision', f'{metrics["precision"]:.4f}'],
            ['Recall', f'{metrics["recall"]:.4f}'],
            ['F1-score', f'{metrics["f1"]:.4f}'],
            ['Log loss', f'{metrics["loss"]:.4f}'],
            ['Вибране α',best['alpha']],
            ['Ітерацій',best['iterations']],
            ['Частка тренування при підборі',f'{best["train_fraction"]*100:.0f}%']])
        cm=metrics['confusion_matrix']
        table(doc, ['Реальний / передбачений','0 — негативний','1 — позитивний'],[
            ['0 — негативний',cm[0][0],cm[0][1]],
            ['1 — позитивний',cm[1][0],cm[1][1]]])
        figure(doc,'training_loss.png','Рис. 1. Динаміка log loss під час фінального навчання.')
        figure(doc,'training_size_vs_accuracy.png','Рис. 2. Максимальна валідаційна точність залежно від обсягу тренування.')
        figure(doc,'confusion_matrix.png','Рис. 3. Матриця помилок на відкладеній тестовій вибірці.')
    else:
        p(doc,'Результати на офіційному IMDb-корпусі ще не отримано. Для підстановки перевірених чисел необхідно запустити `python main.py`, після чого `python make_report.py`. Неперевірені показники тут свідомо не наведено.')
    doc.add_heading('8. Аналіз впливу гіперпараметрів', level=1)
    if metrics:
        with (DIR/'hyperparameters.csv').open(encoding='utf-8-sig', newline='') as file:
            rows=list(csv.DictReader(file))
        top=sorted(rows,key=lambda r:(-float(r['validation_accuracy']),float(r['validation_logloss'])))[:6]
        table(doc,['Частка','α','Ітерації','Валід. accuracy'],[
            [f'{float(r["train_fraction"])*100:.0f}%',r['alpha'],r['iterations'],
             f'{float(r["validation_accuracy"])*100:.2f}%'] for r in top])
        size_best={f:max(float(r['validation_accuracy']) for r in rows if float(r['train_fraction'])==f)
                   for f in (.40,.70,1.0)}
        p(doc,'Найкращі результати за обсягом навчання: '+ '; '.join(
            f'{f*100:.0f}% → {v*100:.2f}%' for f,v in size_best.items())+'.')
        p(doc,'Вплив α та числа ітерацій деталізовано у файлі hyperparameters.csv: більший крок зазвичай пришвидшує збіжність, але надмірний крок може погіршити стабільність; недостатня кількість ітерацій може залишити модель недонавченою. Висновки щодо конкретних чисел слід робити за таблицею фактичних вимірів.')
    else:
        p(doc,'Оцінка буде автоматично сформована за результатами 27 запусків. Очікуваний аналіз: порівняння часток даних, швидкості навчання та кількості ітерацій за метриками на валідації.')
    doc.add_heading('9. Висновки', level=1)
    if metrics:
        p(doc,f'У лабораторній роботі реалізовано бінарну класифікацію тексту засобами логістичної регресії з власним градієнтним спуском. На окремій тестовій вибірці ({metrics["test"]} речень) отримано accuracy {metrics["accuracy"]*100:.2f}% і F1 {metrics["f1"]:.4f}. Під час підбору розглянуто 27 комбінацій гіперпараметрів. Модель придатна для навчальної демонстрації, але її якість залежить від домену та мови тексту; для українських речень потрібні відповідні корпус і обробка.')
    else:
        p(doc,'Реалізовано всі етапи лабораторної роботи та підготовлено процедуру відтворюваного експерименту. Остаточний кількісний висновок можна сформулювати після запуску на реальному корпусі IMDb.')
    doc.add_heading('Використані джерела', level=1)
    p(doc,'1. Kotzias D. Sentiment Labelled Sentences. UCI Machine Learning Repository. 2015. DOI: https://doi.org/10.24432/C57604.')
    p(doc,'2. NLTK: PorterStemmer. https://www.nltk.org/.')
    p(doc,'3. Методичні рекомендації до лабораторної роботи № 1 «Застосування логістичної регресії для аналізу тональності текстів» (наданий викладачем матеріал).')
    doc.save(OUT)
    print('Збережено звіт:',OUT)


if __name__=='__main__':
    if not (DIR/'metrics.json').exists():
        print('Результатів IMDb поки немає. Створюю методичний звіт без вигаданих метрик.')
        build_report(None)
    else:
        build_report(json.loads((DIR/'metrics.json').read_text(encoding='utf-8')))
