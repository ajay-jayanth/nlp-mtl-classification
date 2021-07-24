head_hyper_params = {
    'source': {
        'type': 'BiLSTM',
        'in': 139,
        'H': [40], 
        'out': 2, 
        'LSTM_num_layers': 1, 
        'drop_ps': [0.5, 0.25]
    },
    'audience': {
        'type': 'BiLSTM',
        'in': 139,
        'H': [40], 
        'out': 2, 
        'LSTM_num_layers': 1, 
        'drop_ps': [0.5, 0.25]
    },
    'content': {
        'type': 'BiLSTM',
        'in': 139,
        'H': [40], 
        'out': 3, 
        'LSTM_num_layers': 1, 
        'drop_ps': [0.5, 0.25]
    }
}

if __name__ == '__main__':
    import pandas as pd
    df = pd.DataFrame(head_hyper_params)
    df.to_csv('hyper_configs/hc1.csv')