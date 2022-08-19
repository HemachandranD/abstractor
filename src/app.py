import streamlit as st
import tensorflow as tf
import nltk
import nltk.data
from PIL import Image
import numpy as np
from const import class_names, ordered_class_names
from io import StringIO

st.set_page_config(
    page_title="Abstractor",
    page_icon=":scroll",
    initial_sidebar_state="expanded",
    menu_items={"About": "# This is an *extremely* cool Abstractor app!"},
)


@st.cache(allow_output_mutation=True)
def Load_Abstractor():
    model = tf.keras.models.load_model("src/model/abstractor_model.tflite")
    return model


def setup():
    with st.spinner("Abstractor is being loaded.."):
        model = Load_Abstractor()
        nltk.download("punkt")

    st.write("""# Abstractor 📜""")

    st.write(
        """Abstractor is a Natural Language Processing AI🤖 Web app inspired from Joint sentence Classification Model in Medical abstract that has been trained and fine tuned on top of the Universal Sentence Encoder Deep Neural network."""
    )  # description and instructions

    file = st.file_uploader("Upload the file to be abstracted\n", type=["txt"])
    # file = st.text_area("Please Type or Paste the Medical Abstract to be Skimmed\n")
    # img_file_buffer = st.camera_input("Take a picture")
    return model, file


def preprocess_text(data):
    """Returns a list of dictionaries of abstract line data.

  Takes in filename, reads its contents and sorts through each line,
  extracting things like the target label, the text of the sentence,
  how many sentences are in the current abstract and what sentence number
  the target line is.

  Args:
      filename: a string of the target text file to read and extract line data
      from.

  Returns:
      A list of sentences each containing a line from an abstract
  """
    # Make function to split sentences into characters
    def split_chars(text):
        return " ".join(list(text))

    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    # fp = open(filename)
    # data = fp.read()
    input_lines = tokenizer.tokenize(data)  # get all lines from filename
    sentence = [line for line in input_lines]
    line_number = [sentence.index(line) for line in input_lines]
    chars = [split_chars(line) for line in input_lines]
    return (
        input_lines,
        line_number,
        [(len(input_lines) - 1)] * (len(input_lines)),
        sentence,
        chars,
    )


def upload_predict(file, model):
    """
    Reads in an text file

    Parameters
    ----------
    upload_file (str): string filename of target image
    """
    input_lines, line_number, total_lines, sentence, chars = preprocess_text(file)
    line_number_onehot = tf.one_hot(line_number, depth=15)
    total_lines_onehot = tf.one_hot(total_lines, depth=20)
    input_data = (
        line_number_onehot,
        total_lines_onehot,
        tf.constant(input_lines),
        tf.constant(chars),
    )
    pred_prob = model.predict(input_data)
    predictions = tf.argmax(pred_prob, axis=1)
    # pred_class = class_names[tf.argmax(pred_prob, axis=1)]
    return predictions, pred_prob, sentence


def main():
    model, file = setup()
    if file is None:
        st.text("")
    else:
        # To convert to a string based IO:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        predictions, pred_prob, sentence = upload_predict(string_data, model)
        for i in ordered_class_names:
            for j in class_names:
                if class_names.index(i) == ordered_class_names.index(j):
                    st.write("[" + i + "]")
                    for k in range(0, len(predictions)):
                        if predictions[k] == class_names.index(i):
                            st.write(sentence[k])
                        elif class_names.index(i) not in predictions:
                            st.write("\n")
                else:
                    continue


if __name__ == "__main__":
    main()

# if img_file_buffer is None:
#     st.text("")
# else:
#     image = Image.open(img_file_buffer)
#     st.image(image, use_column_width=True)
#     predictions, pred_prob = upload_predict(image, model)
#     image_class = str(predictions)
#     score = np.round(pred_prob.max() * 100)
#     st.write("This is", image_class)
#     st.write(f"Food Sight is {score}% confident")
