# Source: [3]
import sys
import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import bert_model as bm

def display_pca_scatterplot_3D(embeddings, texts, vocabulary, topn=5, sample=10, highlight=(0,0)):
    """
    assumes vocabulary is a Pandas DateFrame
    Inputs:
        highlight: tuple with start and end index of points that should
            be plotted with a different colour
    """
    print("Computing top 3 PCA components")
    pca_model = PCA(n_components=3, random_state=0)
    three_dim = pca_model.fit_transform(embeddings)#[:,:3]
    print(three_dim.shape)
    print(f"Top 3 PCA components account for a total of {100*np.sum(pca_model.explained_variance_ratio_):.2f} % of the variance in the data")

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []

    first_batch = np.delete(three_dim, np.s_[highlight[0]:highlight[1]], axis=0)
    first_texts = texts[0:highlight[0]] + texts[highlight[1]:]
    print(len(first_batch), len(first_texts))
    trace = go.Scatter3d(
        x = first_batch[:,0],
        y = first_batch[:,1],
        z = first_batch[:,2],
        hovertext = first_texts,
        hoverinfo = "text",
        # text = texts,
        name = "texts",
        textposition = "top center",
        textfont_size = 20,
        mode = 'markers+text',
        marker = {
            'size': 10,
            'opacity': 0.8,
            'color': 2
            }
        )
    data.append(trace)

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    trace_input = go.Scatter3d(
                    x = three_dim[highlight[0]:highlight[1],0],
                    y = three_dim[highlight[0]:highlight[1],1],
                    z = three_dim[highlight[0]:highlight[1],2],
                    text = texts[highlight[0]:highlight[1]],
                    hovertext = first_texts,
                    hoverinfo = "text",
                    name = 'highlighted',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                        }
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )

    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()


if __name__ == '__main__':
    embedding_from_file = False
    texts_from_file = False
    my_bert = bm.BertModel()
    example_usage_string = f"\n\nExample usage: \n{sys.argv[0]} -t text_file.txt\n{sys.argv[0]} -e embedding_file.tsv\n\n"
    if len(sys.argv) == 1:
        print(example_usage_string)
        # txts = ["This is an example text to be embedded.", "Let's see how this works out!", "I'm really excited."]
        #  Draw samples from the embeddings (mostly only English words)
        txts = list(my_bert.get_vocabulary(as_type="df").token[1997:29613].sample(200))
        txts += ["this is a longer string"]
        print(txts)
    elif len(sys.argv) == 2:
        print("\n\nDefault input file treatment: treating input file as input text.\n\n")
        f = sys.argv[2]
        texts_from_file = True
    elif len(sys.argv) == 3:
        if sys.argv[1] == '-t':
            print("\n\nTreating input file as input text.\n\n")
            f = sys.argv[2]
            texts_from_file = True
        elif sys.argv[1] == '-e':
            print("\n\nTreating input file as embeddings.\n\n")
            embedding_from_file = True
            f = sys.argv[2]
            embedding = np.loadtxt(sys.argv[2])
            assert(len(input_embedding.shape) == 2)
    else:
        print(example_usage_string)
        print("\nToo many arguments! Don't know what to do. Exiting.\n")
        sys.exit(0)

    if not embedding_from_file:
        # TODO: Make a function out of this
        highlight = (0,0)
        if texts_from_file:
            txts = [l.strip() for l in open(f, 'r')]
            highlight = (0, len(txts))
            # if len(txts) < 200:
            txts += list(my_bert.get_vocabulary(as_type="df").token[1997:29613].sample(20))
        assert(type(txts) == type([]))
        max_len=max([4] + list(map(len, txts)))
        my_bert.build_model(max_len=max_len)
        embedding = my_bert.get_embedding(txts)
        print(embedding.shape)
    vocab = my_bert.get_vocabulary(as_type="df")
    display_pca_scatterplot_3D(embedding, txts, vocab, highlight=highlight)
