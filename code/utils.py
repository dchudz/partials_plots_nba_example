def test_many_models():
    n_hidden_layers_grid = [2*i for i in range(1, 6)]
    hidden_size_grid = [25, 100, 500, 1000]
    dropout_grid = [0]
    model_info = namedtuple('model_info', ['n_hidden_layers', 'hidden_size', 
                                            'dropout_val', 'fit_history', 'run_time'])
    model_scores = []
    for n_hidden_layers in n_hidden_layers_grid:
        for hidden_size in hidden_size_grid:
            for dropout_val in dropout_grid:
                print(n_hidden_layers, hidden_size, dropout_val)
                start_time = time()
                model = define_nn(n_hidden_layers, hidden_size, dropout_val)
                nn_hist = model.fit(x.values, y.values, nb_epoch=8, batch_size=64, 
                                validation_split=0.3, show_accuracy=True)
                run_time = time() - start_time
                model_scores.append(model_info(n_hidden_layers, hidden_size, dropout_val, nn_hist, run_time))
    output = pd.DataFrame(model_scores, columns=["n_hidden_layers", "hidden_size", "dropout_val", "nn_hist", "run_time"])
    output['best_iteration'] = output.nn_hist.apply(lambda x: np.argmin(x['val_loss']))+1
    output['best_val_score'] = output.nn_hist.apply(lambda x: np.min(x['val_loss']))
    output['best_val_acc'] = output.nn_hist.apply(lambda x: np.max(x['val_acc']))
    output['final_train_score'] = output.nn_hist.apply(lambda x: x['loss'][-1])
    output['final_val_score'] = output.nn_hist.apply(lambda x: x['val_loss'][-1])
    output['train_div_val_score'] = output.final_train_score / output.final_val_score
    return output
