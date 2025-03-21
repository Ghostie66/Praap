import os
import threading
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import seaborn as sns
import sounddevice as sd
import soundfile as sf
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from parselmouth.praat import call
from tabulate import tabulate


# Define screens
class MainMenu(Screen):
    pass


class PitchScreen(Screen):
    pass


class AdvancedScreen(Screen):
    pass


class CompareScreen(Screen):
    pass


# Define KV string
KV = '''
ScreenManager:
    MainMenu:
    PitchScreen:
    AdvancedScreen:
    CompareScreen:

<MainMenu>:
    name: "menu"
    BoxLayout:
        orientation: 'vertical'
        Image:
            id: graph_image
            source: "pitch_graph.png"
        Button:
            text: "Measure Pitch"
            size_hint_y: None
            height: 100
            on_press: app.root.current = "pitch"
        Button:
            text: "Advanced Measurements"
            size_hint_y: None
            height: 100
            on_press: app.root.current = "advanced"
        Button:
            text: "Compare Samples"
            size_hint_y: None
            height: 100
            on_press: app.root.current = "compare"

<PitchScreen>:
    name: "pitch"
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            TextInput:
                id: filename_input
                hint_text: "Enter file name here..."
        Button:
            text: "Record New File"
            on_press: app.audio("pitch")
        Button:
            text: "Analyze"
            on_press: app.process_pitch(filename_input.text)
        Button:
            text: "View Pitch"
            on_press: app.f0_map(filename_input.text,1)
        Button:
            text: "Back to Menu"
            on_press: app.root.current = "menu"

<AdvancedScreen>:
    name: "advanced"
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            TextInput:
                id: filename_input
                hint_text: "Enter file name here..."
            Button:
                text: "Record new file"
                on_press: app.audio("adv")
        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: "Plain Spectrogram"
                on_press: app.draw_spectrogram(filename_input.text)
            Button:
                text: "Intensity Map"
                on_press: app.intensity_spectrogram(filename_input.text)
            Button:
                text: "Fundamental Frequency Map"
                on_press: app.f0_map(filename_input.text,0)
        Button:
            text: "View Intensity Plot"
            on_press: app.draw_intensity(filename_input.text)
        Button:
            text: "View Amplitude"
            on_press: app.draw_amplitude(filename_input.text)
        Button:
            text: "Back to Menu"
            on_press: app.root.current = "menu"

<CompareScreen>:
    name: "compare"
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            TextInput:
                id: beet_input
                hint_text: "Enter the name of a file that only contains the word 'beet'"
            TextInput:
                id: bot_input
                hint_text: "Enter the name of a file that only contains the word 'bot'"
            TextInput:
                id: but_input
                hint_text: "Enter the name of a file that only contains the word 'but'"
        BoxLayout:
            orientation: 'horizontal'
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: "Record yourself saying 'beet'"
                    on_press: app.audio("beet")
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: "Record yourself saying 'bot'"
                    on_press: app.audio("bot")
            BoxLayout:
                orientation: 'vertical'
                Button:
                    text: "Record yourself saying 'but'"
                    on_press: app.audio("but")
        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: "Compare selected files"
                on_press: app.comp(beet_input.text, bot_input.text, but_input.text)
            Button:
                text: "Choose your preferred voice"
                on_press: app.favourite(beet_input.text, bot_input.text, but_input.text)
        Button:
            text: "Back to Menu"
            on_press: app.root.current = "menu"
'''


def save_graph():
    file_path = "data.csv"
    if not os.path.exists(file_path):
        print("No data file found, skipping graph generation.")
        return
    df = pd.read_csv(file_path, names=['Date', 'Value', 'Filename', 'HNR'])
    df['Date'] = df['Date'].astype(str)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    if df.empty:
        print("No valid data to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Value'], marker='o', linestyle='-', color='b', label='Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Data Comparison Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("pitch_graph.png")
    plt.close()
    print("Graph saved successfully.")


class Praapp(App):
    file_path = StringProperty("recording.wav")
    audio_state = StringProperty("ready")
    has_record = False
    recording = False
    audio_buffer = None

    def build(self):
        self.root = Builder.load_string(KV)
        save_graph()
        return self.root

    def process_pitch(self, filename):
        if filename:
            snd = parselmouth.Sound("audio/" + filename)
        else:
            snd = parselmouth.Sound(self.file_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        median = str(round(np.nanmedian(pitch_values)))
        min = str(round(np.nanmin(pitch_values)))
        max = str(round(np.nanmax(pitch_values)))
        file = "data.csv"
        with open(file, "a") as f:
            f.write(f"{date.today()},{np.nanmedian(pitch_values)},{filename},{hnr}\n")
        popup_content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        if hnr < 2:
            hnr_string = "\n Results may not be reliable due to high amounts of noise in the file"
        else:
            hnr_string = ""
        text_popup = Label(text="Your median pitch is {}Hz, with a minimum of {}Hz and a maximum of {}Hz."
                                "\nThe file has a Harmonic-to-noise ratio of {}".format(median, min, max,
                                                                                        round(hnr)) + hnr_string,
                           size_hint_y=None, text_size=(300, None), halign='center', valign='middle')
        popup_content.add_widget(text_popup)
        popup = Popup(title="Pitch Information", content=popup_content, size_hint=(None, None), size=(400, 300))
        popup.open()

    def draw_amplitude(self, filename):
        if filename:
            snd = parselmouth.Sound("audio/" + filename)
        else:
            snd = parselmouth.Sound(self.file_path)
        sns.set()
        plt.figure()
        plt.plot(snd.xs(), snd.values.T)
        plt.xlim([snd.xmin, snd.xmax])
        plt.xlabel("time [s]")
        plt.ylabel("amplitude")
        plt.show()

    def base_spectrogram(self, spectrogram, dynamic_range=70):
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.xlabel("time [s]")
        plt.ylabel("frequency [Hz]")

    def base_intensity(self, intensity):
        plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
        plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
        plt.grid(False)
        plt.ylim(0)
        plt.ylabel("intensity [dB]")

    def draw_spectrogram(self, filename):
        if filename:
            snd = parselmouth.Sound("audio/" + filename)
        else:
            snd = parselmouth.Sound(self.file_path)
        sns.set()
        spectrogram = snd.to_spectrogram()
        plt.figure()
        self.base_spectrogram(spectrogram)
        plt.xlim([snd.xmin, snd.xmax])
        plt.show()

    def intensity_spectrogram(self, filename):
        if filename:
            snd = parselmouth.Sound("audio/" + filename)
        else:
            snd = parselmouth.Sound(self.file_path)
        intensity = snd.to_intensity()
        spectrogram = snd.to_spectrogram()
        plt.figure()
        self.base_spectrogram(spectrogram)
        plt.twinx()
        self.base_intensity(intensity)
        plt.xlim([snd.xmin, snd.xmax])
        plt.show()

    def f0_map(self, filename, f0_type):
        if filename:
            snd = parselmouth.Sound("audio/" + filename)
        else:
            snd = parselmouth.Sound(self.file_path)
        pitch = snd.to_pitch()
        plt.figure()
        if f0_type == 0:
            self.base_spectrogram(snd.to_spectrogram())
            plt.twinx()
        self.draw_f0(pitch)
        plt.xlim([snd.xmin, snd.xmax])
        plt.show()

    def draw_f0(self, pitch):
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, pitch.ceiling)
        plt.ylabel("fundamental frequency [Hz]")

    def draw_intensity(self, filename):
        if filename:
            snd = parselmouth.Sound("audio/" + filename)
        else:
            snd = parselmouth.Sound(self.file_path)
        intensity = snd.to_intensity()
        plt.figure()
        self.base_intensity(intensity)
        plt.xlim([snd.xmin, snd.xmax])
        plt.show()

    def measure_formants(self, sound):
        pitch = sound.to_pitch()
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        if mean_pitch < 120:
            max_formant = 5000
        elif mean_pitch >= 120:
            max_formant = 5500
        else:
            max_formant = 5500  # default/backup value
        sound_formant = sound.to_formant_burg(maximum_formant=max_formant)
        f1 = call(sound_formant, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(sound_formant, "Get mean", 2, 0, 0, "Hertz")
        f3 = call(sound_formant, "Get mean", 3, 0, 0, "Hertz")
        f4 = call(sound_formant, "Get mean", 4, 0, 0, "Hertz")
        return f1, f2, f3, f4

    def comp(self, beet, bot, but):
        if beet and but and bot:
            beet_snd = parselmouth.Sound("audio/" + beet)
            bot_snd = parselmouth.Sound("audio/" + bot)
            but_snd = parselmouth.Sound("audio/" + but)
        else:
            beet_snd = parselmouth.Sound("beet_recording.wav")
            bot_snd = parselmouth.Sound("bot_recording.wav")
            but_snd = parselmouth.Sound("but_recording.wav")
        beetf1, beetf2, beetf3, beetf4 = self.measure_formants(beet_snd)
        botf1, botf2, botf3, botf4 = self.measure_formants(bot_snd)
        butf1, butf2, butf3, butf4 = self.measure_formants(but_snd)
        np.set_printoptions(precision=0)
        headers = ["","Beet", "Bot", "But"]
        arr = np.array([["F1",beetf1,botf1,butf1],["F2",beetf2,botf2,butf2],["F3",beetf3,botf3,butf3],["F4",beetf4,botf4,butf4],["F1-F2 distance",beetf2-beetf1, botf2-botf1,butf2-butf1]])
        table = tabulate(arr, headers, tablefmt="fancy_grid")
        content = Label(text = table)
        popup = Popup(title='Vowel Formant Comparison', content = content, size_hint=(None, None), size=(400, 400))
        popup.open()
        print(table)

    def favourite(self, beet, bot, but):
        popup_content = BoxLayout(orientation='horizontal')
        beet_button = Button(text="Beet", on_press=self.beet_fav)
        bot_button = Button(text="Bot", on_press=self.bot_fav)
        but_button = Button(text="But", on_press=self.but_fav)
        popup_content.add_widget(beet_button)
        popup_content.add_widget(bot_button)
        popup_content.add_widget(but_button)

        popup = Popup(title="Choose your favourite file", content=popup_content, size_hint=(None, None),
                      size=(400, 400))
        popup.open()

    def beet_fav(self, instance):
        beet_info = Label(text="The vowel in 'beet' is produced with the front of the tongue high in the mouth, "
                               "and unrounded lips. Adjusting tongue and lip placement may help your voice sound more "
                               "like this.", size_hint_y=None, text_size=(300, None), halign='center', valign='middle')
        popup = Popup(title="Beet information", content=beet_info, size_hint=(None, None), size=(400, 200))
        popup.open()

    def bot_fav(self, instance):
        bot_info = Label(text="The vowel in 'bot' is produced with the back of the tongue low in the mouth and rounded "
                              "lips. Adjusting tongue and lip placement may help your voice sound more like this.",
                         size_hint_y=None, text_size=(300, None), halign='center', valign='middle')
        popup = Popup(title="Bot information", content=bot_info, size_hint=(None, None), size=(400, 200))
        popup.open()

    def but_fav(self, instance):
        but_info = Label(text="The vowel in 'but' is produced with the middle of the tongue midway between the top and "
                              "bottom of the mouth and unrounded lips. Adjusting tongue and lip placement may help "
                              "your voice sound more like this.", size_hint_y=None, text_size=(370, None),
                         halign='center', valign='middle')
        popup = Popup(title="But information", content=but_info, size_hint=(None, None), size=(450, 200))
        popup.open()

    def audio(self, version):
        match version:
            case "pitch":
                self.file_path = "pitch_recording.wav"
            case "adv":
                self.file_path = "adv_recording.wav"
            case "beet":
                self.file_path = "beet_recording.wav"
            case "bot":
                self.file_path = "bot_recording.wav"
            case "but":
                self.file_path = "but_recording.wav"
            case _:
                self.file_path = "recording.wav"
        popup_content = BoxLayout(orientation='vertical', spacing=20, padding=50)
        state_label = Label(size_hint_y=None, height=40, text='Audio State: ' + self.audio_state)
        location_label = Label(size_hint_y=None, text='Recording Location: ' + self.file_path)
        record_button = Button(text='Start Recording', on_press=self.start_recording)  # Don't call here
        play_button = Button(text='Play', on_press=self.play_recording, disabled=not self.has_record)
        popup_content.add_widget(state_label)
        popup_content.add_widget(location_label)
        popup_content.add_widget(record_button)
        popup_content.add_widget(play_button)
        popup = Popup(title="Pitch Information", content=popup_content, size_hint=(None, None), size=(600, 400))
        popup.open()
        self.state_label = state_label
        self.location_label = location_label
        self.record_button = record_button
        self.play_button = play_button

    def start_recording(self, instance):
        if not self.recording:
            self.audio_state = "recording"
            self.record_button.text = "Press to Stop Recording"
            self.has_record = False
            self.audio_buffer = []
            self.recording = True
            threading.Thread(target=self._record_audio, daemon=True).start()
        else:
            self.audio_state = "ready"
            self.record_button.text = "Start Recording"
            self.recording = False

            if self.audio_buffer:
                self._save_audio()
                self.has_record = True

        self.update_labels()

    def _record_audio(self):
        samplerate = 44100
        channels = 2

        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:
                self.audio_buffer.append(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            while self.recording:
                sd.sleep(100)

    def _save_audio(self):
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)  # Convert list to numpy array
            sf.write(self.file_path, audio_data, 44100)
            print(f"Recording saved to {self.file_path}")

    def play_recording(self, instance):
        if self.has_record:
            self.audio_state = "playing"
            self.play_button.text = "Stop"
            threading.Thread(target=self._play_audio, daemon=True).start()
        else:
            print("No recording available.")

        self.update_labels()

    def _play_audio(self):
        data, samplerate = sf.read(self.file_path)
        sd.play(data, samplerate)
        sd.wait()
        self.audio_state = "ready"
        self.play_button.text = "Play"
        self.update_labels()

    def update_labels(self):
        self.state_label.text = 'Audio State: ' + self.audio_state
        self.play_button.disabled = not self.has_record
        self.record_button.disabled = self.audio_state == "playing"


if __name__ == "__main__":
    Praapp().run()
