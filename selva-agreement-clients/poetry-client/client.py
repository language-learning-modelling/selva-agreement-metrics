from llm_agreement_metrics import dataset, metrics, models
import plotext as plt


def llm_masked_sentences_per_model(
        model, tokenizer,
        masked_sentence_tokens,
        token_idx,
        nMasks,
        ): 
    maskTokenStr=tokenizer.mask_token
    llm_masked_sentences = [
                dataset.generate_llm_masked_sentence(
                    masked_sentence_tokens=masked_sentence_tokens,
                    token_idx=token_idx,
                    nMasks=nMasks,
                    maskTokenStr=maskTokenStr
                ) for nMasks in range(1,4) 
            ]
    return llm_masked_sentences

def main():
    row_tpls = dataset.read_dataset_pandas('./selva_dataset/celvasp_full_annotated_with_metadata_2018_2023.csv')

    cleanedTexts = [ dataset.clean_text(rowText)
                        for (rowVR, rowText) in row_tpls ]
        
    tokenizedTexts = [ dataset.tokenize_text(cleanedText)
                        for cleanedText in cleanedTexts ] 

    models_tpl = models.load_list_of_models([])
    maxNumOfMasks = 3
    for tokenizedText in tokenizedTexts:
        for token_idx, token in enumerate(tokenizedText):
            masked_sentence_tokens = tokenizedText.copy()
            for (model, tokenizer) in models_tpl:
                llm_masked_sentences = \
                    llm_masked_sentences_per_model(
                            model, tokenizer,
                            masked_sentence_tokens,
                            token_idx,
                            nMasks=maxNumOfMasks,
                            )
                llm_masked_sentence = llm_masked_sentences[0]
                sentences_predictions = dataset.fill_masks_pipeline(llm_masked_sentences)
                llm_masked_sentence_predictions = dataset.fill_masks_pipeline([llm_masked_sentence]) 
                k=3
                print(''.join(llm_masked_sentence))
                print(len(llm_masked_sentence_predictions))
                top_k_probs = [str(rag['score']) for rag in llm_masked_sentence_predictions[:k]]
                top_k_str = [rag['token_str'] for rag in llm_masked_sentence_predictions[:k]]
                top_k_labels = [i+1 for i in range(k)]

                plt.simple_bar(top_k_labels,
                               top_k_probs, 
                               width = 100,
                               title = f'model top {k}')
                plt.show() 
                input()
                '''
                for s_idx, sentence_predictions in enumerate(sentences_predictions):
                    k=3
                    print(''.join(llm_masked_sentences[s_idx]))
                    print(len(sentence_predictions))
                    top_k_probs = [rag['score'] for rag in sentence_predictions[:k]]
                    top_k_labels = [i+1 for i in range(k)]

                    plt.simple_bar(top_k_labels,
                                   top_k_probs, 
                                   width = 100,
                                   title = f'model top {k}')
                    plt.show() 
                    input()
                    for p in sentence_predictions:
                        print(f'token: {p["token_str"]} {p["score"]}')
                        print('\n')
                    print(''.join(llm_masked_sentences[s_idx]))
                    input()
                    '''
                # masked_sentence_str = 
                input()
        

if __name__ == '__main__':
    main()
