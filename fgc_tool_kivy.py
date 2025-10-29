import os
import sqlite3
import threading
import requests
import random
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
import math

from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.recycleview import RecycleView
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, ListProperty, NumericProperty, BooleanProperty
from kivy.utils import get_color_from_hex
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window
from PIL import Image as PILImage

# ---------------------------- Country name helpers ----------------------------
name_mapping = {
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
    "China, People's Republic of": "China",
    "Chinese Taipei": "Taiwan",
    "Congo, Democratic Republic of": "Democratic Republic of the Congo",
    "Congo": "Republic of the Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Eswatini": "Eswatini",
    "Great Britain": "United Kingdom",
    "Hong Kong, China": "Hong Kong",
    "Iran, Islamic Republic of": "Iran",
    "Korea, Democratic People's Republic of": "North Korea",
    "Korea, Republic of": "South Korea",
    "Lao People's Democratic Republic": "Laos",
    "Micronesia, Federated States of": "Micronesia",
    "Moldova, Republic of": "Moldova",
    "North Macedonia": "North Macedonia",
    "Russian Federation": "Russia",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Syrian Arab Republic": "Syria",
    "Tanzania, United Republic of": "Tanzania",
    "Timor-Leste": "Timor-Leste",
    "Türkiye": "Turkey",
    "United States of America": "United States",
    "Virgin Islands, British": "British Virgin Islands",
    "Virgin Islands, U.S.": "United States Virgin Islands",
}

hardcoded_countries = {
    "Congo": {"code": "COG", "continent": "Africa", "flag_filename": "Congo.png"},
    "Congo, Democratic Republic of the": {"code": "COD", "continent": "Africa", "flag_filename": "Congo, Democratic Republic of the.png"},
    "Hope": {"code": "", "continent": "Unknown", "flag_filename": None}  # Added for invalid country; review countries.txt
}

# Theme colors used for left/right panels
LEFT_BG = "#ffc8d1"   # pink-ish
RIGHT_BG = "#c5e3f8"  # blue-ish

# ---------------------------- KV Layout ----------------------------
KV = r'''
<TableRow>:
    orientation: 'horizontal'
    size_hint_y: None
    height: '30dp'
    padding: [4, 2]
    spacing: 6
    Label:
        text: root.col1
        color: 1,1,1,1
        text_size: self.size
        halign: 'left'
        valign: 'middle'
    Label:
        text: root.col2
        color: 1,1,1,1
        text_size: self.size
        halign: 'left'
        valign: 'middle'
    Label:
        text: root.col3
        color: 1,1,1,1
        text_size: self.size
        halign: 'left'
        valign: 'middle'
    Label:
        text: root.col4
        color: 1,1,1,1
        text_size: self.size
        halign: 'left'
        valign: 'middle'
    Label:
        text: root.col5
        color: 1,1,1,1
        text_size: self.size
        halign: 'center'
        valign: 'middle'
        
<MainPanel>:
    orientation: 'vertical'
    TabbedPanel:
        id: tabs
        do_default_tab: False
        TabbedPanelItem:
            text: 'Database'
            BoxLayout:
                orientation: 'vertical'
                padding: 8
                spacing: 8
                BoxLayout:
                    size_hint_y: None
                    height: '48dp'
                    spacing: 8
                    Button:
                        text: 'Rebuild Database'
                        on_release: root.on_rebuild_db()
                    Label:
                        id: status_label
                        text: root.status_text
                        size_hint_x: 1
                BoxLayout:
                    size_hint_y: None
                    height: '24dp'
                    ProgressBar:
                        id: progress_bar
                        value: root.progress_value
                        max: root.progress_max
                Label:
                    text: 'Countries Table'
                    size_hint_y: None
                    height: '24dp'
                    bold: True
                    color: 1,1,1,1
                RecycleView:
                    id: rv_table
                    viewclass: 'TableRow'
                    RecycleBoxLayout:
                        default_size: None, dp(30)
                        default_size_hint: 1, None
                        size_hint_y: None
                        height: self.minimum_height
                        orientation: 'vertical'

        TabbedPanelItem:
            text: 'Dashboard'
            ScrollView:
                do_scroll_x: False
                do_scroll_y: True
                bar_width: 10
                BoxLayout:
                    orientation: 'vertical'
                    padding: 16
                    spacing: 16
                    size_hint_y: None
                    height: self.minimum_height

                    # --- Top area: selections + swap toggle ---
                    GridLayout:
                        cols: 3
                        size_hint_y: None
                        height: '180dp'
                        spacing: 20

                        # Left selections
                        BoxLayout:
                            orientation: 'vertical'
                            padding: 8
                            spacing: 6
                            canvas.before:
                                Color:
                                    rgba: 0.2, 0.2, 0.2, 1
                                RoundedRectangle:
                                    pos: self.pos
                                    size: self.size
                                    radius: [8]
                            Label:
                                text: 'Left Selections'
                                size_hint_y: None
                                height: '20dp'
                                bold: True
                            BoxLayout:
                                id: left_spinners
                                orientation: 'vertical'
                                spacing: 8

                        # Right selections
                        BoxLayout:
                            orientation: 'vertical'
                            padding: 8
                            spacing: 6
                            canvas.before:
                                Color:
                                    rgba: 0.2, 0.2, 0.2, 1
                                RoundedRectangle:
                                    pos: self.pos
                                    size: self.size
                                    radius: [8]
                            Label:
                                text: 'Right Selections'
                                size_hint_y: None
                                height: '20dp'
                                bold: True
                            BoxLayout:
                                id: right_spinners
                                orientation: 'vertical'
                                spacing: 8

                        # Color swap toggle
                        BoxLayout:
                            orientation: 'vertical'
                            padding: 10
                            spacing: 10
                            size_hint_x: None
                            width: '160dp'
                            canvas.before:
                                Color:
                                    rgba: 0.2, 0.2, 0.2, 1
                                RoundedRectangle:
                                    pos: self.pos
                                    size: self.size
                                    radius: [8]
                            Label:
                                text: 'Swap Colors'
                                size_hint_y: None
                                height: '20dp'
                            CheckBox:
                                id: swap_cb
                                active: root.swap_colors
                                size_hint_y: None
                                height: '40dp'
                                on_active: root.toggle_swap(self.active)

                    # --- Buttons ---
                    BoxLayout:
                        orientation: 'horizontal'
                        size_hint_y: None
                        height: '50dp'
                        spacing: 20
                        Button:
                            text: 'Clear Information'
                            on_release: root.clear_info()

                    # --- Country Info Panels ---
                    BoxLayout:
                        id: panels_container
                        orientation: 'horizontal'
                        size_hint_y: None
                        height: self.minimum_height
                        spacing: 16
                        padding: [10, 10, 10, 10]

                        BoxLayout:
                            id: left_column
                            orientation: 'vertical'
                            size_hint_x: 0.5
                            size_hint_y: None
                            height: self.minimum_height
                            spacing: 12

                        BoxLayout:
                            id: right_column
                            orientation: 'vertical'
                            size_hint_x: 0.5
                            size_hint_y: None
                            height: self.minimum_height
                            spacing: 12

        TabbedPanelItem:
            text: 'Flashcard'
            BoxLayout:
                orientation: 'vertical'
                padding: 8
                spacing: 8
                Image:
                    id: flash_img
                    size_hint_y: 0.6
                    allow_stretch: True
                    keep_ratio: True
                BoxLayout:
                    id: flash_info
                    orientation: 'vertical'
                    size_hint_y: None
                    height: '160dp'
                    spacing: dp(4)
                    padding: [dp(4), dp(4)]
                BoxLayout:
                    size_hint_y: None
                    height: '48dp'
                    spacing: 8
                    Button:
                        text: 'Show Random Flag'
                        on_release: root.on_flash_random()
                    Button:
                        text: 'Reveal Information'
                        on_release: root.on_flash_reveal()

        TabbedPanelItem:
            text: 'Reference'
            FloatLayout:
                Label:
                    id: ref_missing
                    text: ''
                    size_hint: None, None
                    size: self.texture_size
                    pos_hint: {'center_x': 0.5, 'top': 1}
                    color: 1, 0, 0, 1
                Image:
                    id: ref_img
                    size_hint: 1, 1
                    allow_stretch: True
                    keep_ratio: True
                    pos_hint: {'center_x': 0.5, 'center_y': 0.5}

'''

# ---------------------------- TableRow viewclass ----------------------------
class TableRow(BoxLayout):
    col1 = StringProperty('')
    col2 = StringProperty('')
    col3 = StringProperty('')
    col4 = StringProperty('')
    col5 = StringProperty('')

# ---------------------------- ComboBox widget ----------------------------
class ComboBox(BoxLayout):
    """TextInput with anchored, filterable DropDown (no extra button). Optimized: Debounced open/filter."""
    __events__ = ('on_select',)

    text = StringProperty('')
    placeholder = StringProperty('select country…')
    options = ListProperty([])
    auto_open = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, height=dp(36))
        self._debounce_clock = None  # For debouncing

        # Editor
        self._ti = TextInput(text='', hint_text=self.placeholder, multiline=False, write_tab=False,
                             size_hint_y=None, height=dp(36), padding=[dp(8), dp(8), dp(8), dp(8)])
        self._ti.bind(text=self._on_text_change, focus=self._on_focus, on_text_validate=self._on_validate,
                      on_touch_down=self._on_ti_touch)
        self.add_widget(self._ti)

        # DropDown menu
        self._dd = DropDown(auto_dismiss=True, max_height=dp(280))
        self._filtered = []

    def on_select(self, *args):
        pass

    # ----- Events & open logic (debounced) -----
    def _on_focus(self, _ti, focused: bool):
        if focused and self.auto_open:
            self._debounce_open(0)

    def _on_ti_touch(self, instance, touch):
        if instance.collide_point(*touch.pos) and instance.focus:
            self._debounce_open(0)
        return False

    def _debounce_open(self, delay=0.1):
        if self._debounce_clock:
            self._debounce_clock.cancel()
        self._debounce_clock = Clock.schedule_once(lambda dt: self.open(), delay)

    def _on_text_change(self, _ti, val):
        self.text = val
        if self._dd.attach_to:
            self._apply_filter(val)  # Apply immediately if open, else on open

    def _on_validate(self, *_):
        if self._filtered:
            self._choose(self._filtered[0])

    # ----- Filter & build -----
    def _apply_filter(self, q):
        ql = (q or '').strip().lower()
        self._filtered = [o for o in self.options if ql in o.lower()] if ql else list(self.options)
        self._dd.clear_widgets()
        if not self._filtered:
            no_btn = Button(text='No matches', size_hint_y=None, height=dp(36), disabled=True)
            self._dd.add_widget(no_btn)
        else:
            for opt in self._filtered:
                btn = Button(text=opt, size_hint_y=None, height=dp(36))
                btn.bind(on_release=lambda b: self._choose(b.text))
                self._dd.add_widget(btn)

    def _choose(self, val):
        self.text = val
        self._ti.text = val
        self._dd.dismiss()
        self.dispatch('on_select')

    def open(self):
        # Rebuild and open anchored to the TextInput; schedule avoids immediate dismiss by the same touch
        self._apply_filter(self._ti.text)
        Clock.schedule_once(lambda dt: self._dd.open(self._ti), 0)

    def set_options(self, items):
        self.options = sorted(list(items))  # Pre-sort for faster filtering
        if self._dd.attach_to:
            self._apply_filter(self._ti.text)

# ---------------------------- MainPanel ----------------------------
class MainPanel(BoxLayout):
    status_text = StringProperty("Click 'Rebuild Database' to start")
    progress_value = NumericProperty(0)
    progress_max = NumericProperty(1)
    swap_colors = BooleanProperty(False)
    _show_info_debounce = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_file = 'flags.db'
        self.txt_file = 'countries.txt'
        Clock.schedule_once(self.post_init, 0.1)

    def post_init(self, dt):
        # Add six ComboBoxes
        left_box = self.ids.left_spinners
        right_box = self.ids.right_spinners
        def make_cb():
            cb = ComboBox(placeholder='select country…')
            cb.bind(on_select=lambda *_: self._debounce_show_info(0.1))
            return cb
        for _ in range(3): left_box.add_widget(make_cb())
        for _ in range(3): right_box.add_widget(make_cb())

        # Reference image block
        ref_path = 'fgc_EE_ref.png'
        if not os.path.exists(ref_path):
            self.ids.ref_missing.text = f"Image not found: {ref_path}"
            self.ids.ref_img.source = ''
        else:
            self.ids.ref_missing.text = ''
            self.ids.ref_img.source = ref_path

        # Ensure DB
        if not os.path.exists(self.db_file) or not self.check_table_exists(self.db_file):
            self.on_rebuild_db()
        else:
            self.refresh_table()
            self.update_dropdowns()

    # ---------- DB helpers ----------
    def check_table_exists(self, db_file):
        try:
            with sqlite3.connect(db_file) as conn:
                conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL for better concurrency
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='countries'")
                return cur.fetchone() is not None
        except Exception as e:
            print(f"check_table_exists error: {e}")
            return False

    def on_rebuild_db(self):
        print("Starting database rebuild...")
        threading.Thread(target=self.build_database, args=(self.txt_file,), daemon=True).start()

    def _fetch_country_data(self, country, max_retries=3):
        """Fetch API data with retries."""
        print(f"Fetching data for {country}...")
        query_name = name_mapping.get(country, country)
        safe_name = country.replace('/', '_').replace('\\', '_')
        expected_path = os.path.join('flags', f"{safe_name}.png")

        for attempt in range(max_retries):
            try:
                resp = requests.get(f"https://restcountries.com/v3.1/name/{query_name}?fullText=true", timeout=5)
                print(f"API response for {country}: status={resp.status_code}")
                if resp.status_code == 200 and resp.json():
                    candidates = resp.json()
                    data = None
                    for c in candidates:
                        if c.get('name', {}).get('official', '').lower() == country.lower():
                            data = c; break
                    if not data:
                        data = candidates[0]
                    # Download flag if not exists
                    if not os.path.exists(expected_path):
                        png_url = data.get('flags', {}).get('png')
                        if png_url:
                            r = requests.get(png_url, timeout=5)
                            if r.status_code == 200:
                                img = PILImage.open(BytesIO(r.content)).convert('RGBA')
                                img = img.resize((600, 360), PILImage.LANCZOS)
                                img.save(expected_path, format='PNG')
                                print(f"Saved flag for {country} at {expected_path}")
                            else:
                                print(f"Flag download failed for {country}: Status {r.status_code}")
                                return None, None
                        else:
                            print(f"No flag URL for {country}")
                            return None, None
                    # Extract continent and code
                    continent = data.get('continents', [None])[0] or 'Unknown'
                    code = data.get('cca3', '') or ''
                    print(f"Extracted for {country}: continent={continent}, code={code}")
                    return data, expected_path
                else:
                    print(f"API call failed for {country}: Status {resp.status_code}, Response: {resp.text[:100]}...")
                    return None, None
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"API error for {country} after {max_retries} attempts: {e}")
                    return None, None
                time.sleep(2 ** attempt)  # Exponential backoff

    def build_database(self, file_path):
        try:
            print(f"Reading countries from {file_path}...")
            if not os.path.exists(file_path):
                Clock.schedule_once(lambda dt: self._set_status(f"Error: {file_path} not found"), 0)
                print(f"Error: {file_path} not found")
                return
            with open(file_path, 'r', encoding='utf-8') as f:
                countries = []
                seen = set()
                for line in f:
                    line = line.strip()
                    if line and line not in ('Top of Form', 'Bottom of Form', 'zzz'):
                        parts = line.split('|')
                        country = parts[0].strip(); pron = parts[1].strip() if len(parts) > 1 else ''
                        if country not in seen:
                            countries.append((country, pron))
                            seen.add(country)
                        else:
                            print(f"Skipping duplicate country: {country}")
                countries = sorted(countries, key=lambda x: x[0])
            if not countries:
                Clock.schedule_once(lambda dt: self._set_status('Error: No countries found!'), 0)
                print("Error: No countries found in countries.txt")
                return
            print(f"Found {len(countries)} unique countries in {file_path}")

            os.makedirs('flags', exist_ok=True)
            total = len(countries)
            Clock.schedule_once(lambda dt: self._set_progress_max(total), 0)

            # Hardcoded first
            hardcoded_data = []
            api_countries = []
            skipped_countries = []
            print("Processing hardcoded countries...")
            for i, (country, pron) in enumerate(countries):
                if country in hardcoded_countries:
                    info = hardcoded_countries[country]
                    code = info['code']; continent = info['continent']
                    flag_filename = info['flag_filename']
                    expected_path = os.path.join('flags', flag_filename) if flag_filename else None
                    flag_path = expected_path if expected_path and os.path.exists(expected_path) else None
                    hardcoded_data.append((country, continent, pron, flag_path, code))
                    print(f"Added hardcoded country: {country}")
                else:
                    api_countries.append((country, pron, i))  # Track original index

            # Parallel API fetches
            print(f"Starting API fetches for {len(api_countries)} countries...")
            api_data = {c: None for c, _ , _ in api_countries}
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self._fetch_country_data, c): (c, p) for c, p, _ in api_countries}
                completed = 0
                for future in as_completed(futures):
                    country, pron = futures[future]
                    try:
                        data, flag_path = future.result()
                        if data is None or flag_path is None:
                            print(f"Skipping {country}: No valid data or flag available")
                            skipped_countries.append(country)
                            continue
                        continent = data.get('continents', [None])[0]
                        if continent is None:
                            print(f"Skipping {country}: No continent data")
                            skipped_countries.append(country)
                            continue
                        code = data.get('cca3', '')
                        if not code:
                            print(f"Skipping {country}: No CCA3 code")
                            skipped_countries.append(country)
                            continue
                        api_data[country] = (continent, pron, flag_path, code)
                        print(f"Processed {country}: continent={continent}, code={code}")
                    except Exception as e:
                        print(f"Future error for {country}: {e}")
                        skipped_countries.append(country)
                        api_data[country] = None
                    completed += 1
                    if completed % 10 == 0:  # Batch progress
                        Clock.schedule_once(lambda dt, v=completed: self._set_progress(v), 0)

            # Combine and insert only valid data
            all_data = hardcoded_data + [
                (c, *api_data[c]) for c in [cc[0] for cc in api_countries] if api_data[c] is not None
            ]
            print(f"Prepared {len(all_data)} valid entries for database insertion")
            if not all_data:
                Clock.schedule_once(lambda dt: self._set_status("Error: No valid country data to insert"), 0)
                print("Error: No valid country data to insert")
                return

            print("Writing to database...")
            try:
                with sqlite3.connect(self.db_file) as conn:
                    conn.execute('PRAGMA journal_mode=WAL')
                    cur = conn.cursor()
                    cur.execute('''CREATE TABLE IF NOT EXISTS countries
                                   (name TEXT PRIMARY KEY, continent TEXT,
                                    pronunciation TEXT, flag_path TEXT, code TEXT)''')
                    cur.execute('DELETE FROM countries')
                    cur.executemany('''INSERT OR REPLACE INTO countries
                                    (name, continent, pronunciation, flag_path, code)
                                    VALUES (?, ?, ?, ?, ?)''', all_data)
                    conn.commit()
                    # Verify insertion
                    cur.execute('SELECT COUNT(*) FROM countries')
                    row_count = cur.fetchone()[0]
                    print(f"Inserted {row_count} rows into database")
                    if row_count != len(all_data):
                        print(f"Warning: Expected {len(all_data)} rows, but inserted {row_count}")
            except sqlite3.Error as e:
                Clock.schedule_once(lambda dt: self._set_status(f"Database error: {e}"), 0)
                print(f"Database insertion error: {e}")
                return

            skip_msg = f" Skipped {len(skipped_countries)} invalid countries: {', '.join(skipped_countries)}" if skipped_countries else ""
            Clock.schedule_once(lambda dt: self._set_status(f"Database build complete! {row_count} countries.{skip_msg}"), 0)
            Clock.schedule_once(lambda dt: (self.update_dropdowns() or self.refresh_table()), 1.0)
            print(f"Database build complete: {row_count} countries inserted. {skip_msg}")
        except Exception as e:
            Clock.schedule_once(lambda dt: self._set_status(f"Error: {e}"), 0)
            print(f"Build error: {e}")

    # ---------- UI property helpers ----------
    def _set_status(self, text):
        self.status_text = text
        print(f"Status updated: {text}")

    def _set_progress(self, value):
        self.progress_value = value
        print(f"Progress updated: {value}/{self.progress_max}")

    def _set_progress_max(self, v):
        self.progress_max = max(1, v)
        self.progress_value = 0
        print(f"Progress max set to: {v}")

    # ---------- Table & dropdowns ----------
    def refresh_table(self):
        try:
            print("Refreshing table...")
            with sqlite3.connect(self.db_file) as conn:
                cur = conn.cursor()
                cur.execute('SELECT name, continent, pronunciation, flag_path, code FROM countries ORDER BY name')
                rows = cur.fetchall()
            data = [
                {'col1': n or '', 'col2': c or '', 'col3': p or '', 'col4': fp or '', 'col5': cd or ''}
                for n, c, p, fp, cd in rows
            ]
            self.ids.rv_table.data = data
            print(f"Table refreshed with {len(data)} rows")
            self._set_progress(0)
        except Exception as e:
            self._set_status(f"DB error: {e}")
            print(f"refresh_table error: {e}")

    def update_dropdowns(self):
        try:
            print("Updating dropdowns...")
            with sqlite3.connect(self.db_file) as conn:
                cur = conn.cursor()
                cur.execute('SELECT name FROM countries ORDER BY name')
                countries = [r[0] for r in cur.fetchall()]
            for w in list(self.ids.left_spinners.children) + list(self.ids.right_spinners.children):
                if isinstance(w, ComboBox):
                    w.set_options(countries)
            print(f"Dropdowns updated with {len(countries)} countries")
        except Exception as e:
            print(f"update_dropdowns error: {e}")

    def _debounce_show_info(self, delay=0.1):
        if self._show_info_debounce:
            self._show_info_debounce.cancel()
        self._show_info_debounce = Clock.schedule_once(lambda dt: self.show_info(), delay)

    def toggle_swap(self, active):
        self.swap_colors = active
        self._debounce_show_info(0)  # Immediate repaint

    def _read_country_from_widget(self, w):
        txt = getattr(w, 'text', '') or ''
        return txt if txt and txt != '[select country]' else ''

    def clear_info(self):
        """Clear both info columns and reset all ComboBoxes to empty text."""
        self.ids.left_column.clear_widgets()
        self.ids.right_column.clear_widgets()
        for w in list(self.ids.left_spinners.children) + list(self.ids.right_spinners.children):
            if isinstance(w, ComboBox):
                w._ti.text = ''
                w.text = ''
                if w._dd and w._dd.attach_to:
                    w._dd.dismiss()
        if 'flash_info' in self.ids:
            self.ids.flash_info.clear_widgets()
        if 'flash_img' in self.ids:
            self.ids.flash_img.source = ''

    def calculate_text_box_height(self, text_box):
        """Calculate the natural height of a text_box based on its child labels."""
        for lbl in text_box.children:
            if hasattr(lbl, '_updating'):
                lbl._updating = True
                try:
                    lbl.text_size = (lbl.width, None)
                    lbl.texture_update()
                    new_height = max(lbl.height, lbl.texture_size[1] + dp(6))
                    lbl.height = new_height
                finally:
                    lbl._updating = False
        total_height = sum(c.height for c in text_box.children) + text_box.spacing * (len(text_box.children) - 1) + sum(text_box.padding[1::2])
        return max(dp(180), total_height)

    def _add_country_panel(self, container, country, code, continent, pron, flag_path, bg_color, is_right_column=False):
        from kivy.utils import get_color_from_hex

        # Base panel with dynamic height
        panel = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(200),  # Minimum height
            padding=[dp(8), dp(8)],
            spacing=dp(10)
        )
        with panel.canvas.before:
            Color(*get_color_from_hex(bg_color))
            rect = RoundedRectangle(pos=panel.pos, size=panel.size, radius=[10])
        panel.bind(pos=lambda _, v: setattr(rect, 'pos', v))
        panel.bind(size=lambda _, v: setattr(rect, 'size', v))

        # Darker background color for code text
        r, g, b, a = get_color_from_hex(bg_color)
        dark_factor = 0.45
        dark_color = (r * dark_factor, g * dark_factor, b * dark_factor, 1)

        # Text box with dynamic height
        text_box = BoxLayout(
            orientation='vertical',
            size_hint_x=0.55,
            size_hint_y=None,
            height=dp(180),  # Initial minimum height
            spacing=dp(4),
            padding=[dp(4), dp(4)],
            pos_hint={'center_y': 0.5}
        )

        # Flag image with dynamic height to match text_box
        img = Image(
            source=flag_path if flag_path and os.path.exists(flag_path) else '',
            allow_stretch=True,
            keep_ratio=True,
            size_hint_x=0.45,
            size_hint_y=None,
            height=dp(180)  # Initial height, will sync with text_box
        )

        # Helper to create labels with fixed height for code and continent
        def make_fixed_label(text, font_size, color, bold=False, height=dp(25)):
            lbl = Label(
                text=text,
                font_size=font_size,
                bold=bold,
                color=color,
                halign='center',
                size_hint_y=None,
                height=height
            )
            lbl.bind(
                size=lambda lbl, _: setattr(lbl, 'text_size', (lbl.width, None))
            )
            return lbl

        # Helper to create labels with dynamic height and font size for name and pronunciation
        def make_dynamic_label(text, font_size, color, bold=False, min_height=dp(25)):
            lbl = Label(
                text=text,
                font_size=font_size,
                bold=bold,
                color=color,
                halign='center',
                size_hint_y=None,
                height=min_height
            )
            lbl._updating = False
            def adjust_height_and_font(*_):
                if lbl._updating:
                    return
                lbl._updating = True
                try:
                    # Set text_size to allow wrapping
                    lbl.text_size = (lbl.width, None)
                    lbl.texture_update()
                    # Base font size based on label type
                    base_font_size = sp(36) if text == country else sp(20)
                    wrapped_font_size = sp(28) if text == country else sp(16)
                    # Check if text has no spaces
                    no_spaces = text.find(' ') == -1
                    # Estimate single-line height
                    single_line_height = base_font_size + dp(6)
                    # Check if text would wrap
                    is_wrapped = lbl.texture_size[1] > single_line_height
                    if no_spaces and lbl.texture_size[0] > lbl.text_size[0]:
                        # Shrink font to fit width, no wrapping
                        scale = lbl.text_size[0] / lbl.texture_size[0]
                        new_font_size = max(sp(12), base_font_size * scale)
                        lbl.font_size = new_font_size
                        lbl.text_size = (lbl.width, base_font_size + dp(6))  # Force single line
                        lbl.texture_update()
                        new_height = max(min_height, lbl.texture_size[1] + dp(6))
                    else:
                        # Normal wrapping behavior
                        lbl.font_size = wrapped_font_size if is_wrapped else base_font_size
                        lbl.text_size = (lbl.width, None)
                        lbl.texture_update()
                        new_height = max(min_height, lbl.texture_size[1] + dp(6))
                    lbl.height = new_height
                finally:
                    lbl._updating = False
            lbl.bind(size=adjust_height_and_font, texture_size=adjust_height_and_font)
            Clock.schedule_once(lambda dt: adjust_height_and_font(), 0.1)  # Initial adjustment
            return lbl

        # Create labels
        code_label = make_fixed_label(code or '', '36sp', dark_color, bold=True, height=dp(48))
        name_label = make_dynamic_label(country, '48sp', (0, 0, 0, 1), bold=True, min_height=dp(40))
        continent_label = make_fixed_label(continent, '20sp', (0, 0, 0, 1), height=dp(25))
        pron_label = make_dynamic_label(f"Pronunciation: {pron}" if pron else '', '20sp', (0, 0, 0, 1), min_height=dp(25))

        text_box.add_widget(code_label)
        text_box.add_widget(name_label)
        text_box.add_widget(continent_label)
        text_box.add_widget(pron_label)

        # Update text_box, img, and panel heights
        def update_text_box_height(text_box, img, panel, height=None):
            if hasattr(text_box, '_updating') and text_box._updating:
                return
            text_box._updating = True
            try:
                # Use provided height or calculate natural height
                new_height = height if height is not None else self.calculate_text_box_height(text_box)
                text_box.height = new_height
                img.height = new_height
                panel.height = max(dp(200), new_height + 2 * panel.padding[1])
            finally:
                text_box._updating = False

        # Bind child label heights to trigger update
        for child in text_box.children:
            child.bind(height=lambda *_: update_text_box_height(text_box, img, panel))

        # Initial height update
        Clock.schedule_once(lambda dt: update_text_box_height(text_box, img, panel), 0.2)

        # Layout order
        if is_right_column:
            panel.add_widget(text_box)
            panel.add_widget(img)
        else:
            panel.add_widget(img)
            panel.add_widget(text_box)
        container.add_widget(panel)
        return panel, text_box, img, partial(update_text_box_height, text_box, img, panel)

    def show_info(self):
        left_col = self.ids.left_column
        right_col = self.ids.right_column
        left_col.clear_widgets()
        right_col.clear_widgets()

        # Determine colors based on toggle
        left_bg = LEFT_BG
        right_bg = RIGHT_BG
        if self.swap_colors:
            left_bg, right_bg = right_bg, left_bg

        left_panels = []
        right_panels = []
        try:
            with sqlite3.connect(self.db_file) as conn:
                cur = conn.cursor()
                # Batch left queries
                left_countries = [self._read_country_from_widget(w) for w in self.ids.left_spinners.children[::-1] if self._read_country_from_widget(w)]
                if left_countries:
                    placeholders = ','.join(['?' for _ in left_countries])
                    cur.execute(f'SELECT name, continent, pronunciation, flag_path, code FROM countries WHERE name IN ({placeholders})', left_countries)
                    left_rows = {row[0]: row[1:] for row in cur.fetchall()}
                    for country in left_countries:
                        if country in left_rows:
                            continent, pron, flag_path, code = left_rows[country]
                            panel, text_box, img, update_fn = self._add_country_panel(left_col, country, code, continent, pron, flag_path, left_bg, False)
                            left_panels.append((panel, text_box, img, update_fn))

                # Batch right queries
                right_countries = [self._read_country_from_widget(w) for w in self.ids.right_spinners.children[::-1] if self._read_country_from_widget(w)]
                if right_countries:
                    placeholders = ','.join(['?' for _ in right_countries])
                    cur.execute(f'SELECT name, continent, pronunciation, flag_path, code FROM countries WHERE name IN ({placeholders})', right_countries)
                    right_rows = {row[0]: row[1:] for row in cur.fetchall()}
                    for country in right_countries:
                        if country in right_rows:
                            continent, pron, flag_path, code = right_rows[country]
                            panel, text_box, img, update_fn = self._add_country_panel(right_col, country, code, continent, pron, flag_path, right_bg, True)
                            right_panels.append((panel, text_box, img, update_fn))

            # Synchronize panel heights per row
            def synchronize_heights(dt=None):
                max_rows = max(len(left_panels), len(right_panels))
                for row_idx in range(max_rows):
                    pair = []
                    if row_idx < len(left_panels):
                        pair.append(left_panels[row_idx])
                    if row_idx < len(right_panels):
                        pair.append(right_panels[row_idx])
                    if len(pair) > 1:
                        # Calculate natural heights for each panel
                        text_heights = []
                        for _, text_box, _, _ in pair:
                            height = self.calculate_text_box_height(text_box)
                            text_heights.append(height)
                        # Get max height for the row
                        max_text_height = max(text_heights)
                        # Apply max height to both panels
                        for _, _, _, update_fn in pair:
                            update_fn(max_text_height)
                    elif pair:
                        # Handle unpaired panel
                        _, text_box, _, update_fn = pair[0]
                        height = self.calculate_text_box_height(text_box)
                        update_fn(height)

            # Initial synchronization
            Clock.schedule_once(synchronize_heights, 0.7)

            # Bind to window size and label heights for dynamic updates
            def on_resize_or_label_change(*_):
                Clock.schedule_once(synchronize_heights, 0.2)

            Window.bind(on_resize=on_resize_or_label_change)
            for _, text_box, _, _ in left_panels + right_panels:
                for child in text_box.children:
                    child.bind(height=on_resize_or_label_change)

        except Exception as e:
            print('show_info error:', e)

    # ---------- Flashcard ----------
    def on_flash_random(self):
        try:
            with sqlite3.connect(self.db_file) as conn:
                cur = conn.cursor()
                cur.execute('SELECT name, flag_path FROM countries ORDER BY RANDOM() LIMIT 1')
                row = cur.fetchone()
            if row:
                self._current_country, path = row
                if path and os.path.exists(path):
                    self.ids.flash_img.source = path
                    self.ids.flash_info.clear_widgets()
                else:
                    self.ids.flash_img.source = ''
                    self.ids.flash_info.clear_widgets()
                    self.ids.flash_info.add_widget(Label(text='Flag image missing', font_size='20sp', color=(1, 1, 1, 1), halign='center', valign='middle', size_hint_y=None, height=dp(30)))
            else:
                self.ids.flash_info.clear_widgets()
                self.ids.flash_info.add_widget(Label(text='No data', font_size='20sp', color=(1, 1, 1, 1), halign='center', valign='middle', size_hint_y=None, height=dp(30)))
        except Exception as e:
            print('flash random error:', e)
            self.ids.flash_info.clear_widgets()
            self.ids.flash_info.add_widget(Label(text=f'Error: {e}', font_size='20sp', color=(1, 1, 1, 1), halign='center', valign='middle', size_hint_y=None, height=dp(30)))

    def on_flash_reveal(self):
        try:
            if not hasattr(self, '_current_country') or not self._current_country:
                return
            with sqlite3.connect(self.db_file) as conn:
                cur = conn.cursor()
                cur.execute('SELECT continent, pronunciation, code FROM countries WHERE name=?', (self._current_country,))
                row = cur.fetchone()
            if row:
                continent, pron, code = row
                self.ids.flash_info.clear_widgets()

                # Helper to create labels matching dashboard styling with white text
                def make_label(text, font_size, bold=False):
                    lbl = Label(
                        text=text,
                        font_size=font_size,
                        bold=bold,
                        color=(1, 1, 1, 1),
                        halign='center',
                        valign='middle',
                        size_hint_y=None
                    )
                    lbl.bind(
                        size=lambda lbl, _: setattr(lbl, 'text_size', (lbl.width, None)),
                        texture_size=lambda lbl, ts: setattr(lbl, 'height', ts[1] + dp(6))
                    )
                    return lbl

                # Create labels with dashboard styling
                code_label = make_label(code, '36sp', bold=True)
                name_label = make_label(self._current_country, '30sp', bold=True)
                info_text = f"{continent}\nPronunciation: {pron}" if pron else continent
                info_label = make_label(info_text, '20sp')

                # Add labels to flash_info
                self.ids.flash_info.add_widget(code_label)
                self.ids.flash_info.add_widget(name_label)
                self.ids.flash_info.add_widget(info_label)
            else:
                self.ids.flash_info.clear_widgets()
                self.ids.flash_info.add_widget(Label(text='No data', font_size='20sp', color=(1, 1, 1, 1), halign='center', valign='middle', size_hint_y=None, height=dp(30)))
        except Exception as e:
            self.ids.flash_info.clear_widgets()
            self.ids.flash_info.add_widget(Label(text=f'Error: {e}', font_size='20sp', color=(1, 1, 1, 1), halign='center', valign='middle', size_hint_y=None, height=dp(30)))

# ---------------------------- App ----------------------------
class FGCApp(App):
    def build(self):
        self.title = 'FGC GA Tool'
        self.icon = 'fgc.ico'
        #Window.size = (1000, 700)
        Builder.load_string(KV)
        return MainPanel()

if __name__ == '__main__':
    FGCApp().run()