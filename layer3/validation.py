from .model import WorkforceModel

if __name__ == '__main__':
    scenario = {
        'tool_complexity'  : 0.65,
        'support_model'    : 'chatbot',
        'manager_signal'   : 0.40,
        'training_intensity': 0.10
    }
 
    print('Loading model...')
    m = WorkforceModel(scenario, rng=42)
    print(f'Agents loaded: {len(m.agents)}')   # expect 1000
 
    print('Running 5 weeks...')
    for _ in range(5):
        m.step()
 
    results = m.datacollector.get_model_vars_dataframe()
    print('\n--- Results (5 weeks) ---')
    print(results.to_string())
 
    print('\n--- Validation checks ---')
    assert len(results) == 5,                              'FAIL: should have 5 rows'
    assert not results.isnull().any().any(),               'FAIL: NaN values found'
    assert 0.10 < results['adoption_rate'].iloc[0] < 0.50, 'FAIL: week-0 adoption rate out of range'
    assert results['ticket_volume'].iloc[1] > 0,           'FAIL: no tickets at week 1'
    assert results['exs_score'].iloc[0] > 0,               'FAIL: EXS score is zero'
 
    print('All checks passed — model.py is working')