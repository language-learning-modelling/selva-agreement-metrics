def plot(models_fps, top_k_probs, top_k_str):
    '''
    given a list of models that for each you have
    a list of probabilities plot their probabilities
    grouped by token 

    plt.simple_multiple_bar(top_k_str,
                   top_k_probs,
                   width= 0.1,
                   )
    plt.title(f'model top {k}')
    plt.show() 
    '''
    models_names = [
            m.replace('c4200m','full-efcamdat').split('/')[-1][5:-15] 
                if len(m) > 50
                else 
            m.replace('c4200m','full-efcamdat').split('/')[-1] 
                for m in models_fps
            ]
    model_names = [
            'learner-model-full-efcamdat',
            'native-model-bert'
            ]
    tokens = top_k_str 
    probs_per_model = {
        model_name : probs_lst
            for model_name, probs_lst
                in zip(models_names,
                        top_k_probs) 
    }

    x = np.arange(len(tokens))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in probs_per_model.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Model probabilities by tokens')
    ax.set_xticks(x + width, tokens)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.show()
    input()

