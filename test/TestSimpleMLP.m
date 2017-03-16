function tests = TestSimpleMLP
    tests = functiontests( localfunctions );
end

function testUnion( testCase )
% Requier Statistic and Machine Learning Toolbox
    addpath( '../' );
    addpath( '../utilities' );
    
    dataset = load( '../datasets/iris_scaled.csv' );
    
    num_instances = size( dataset, 1 );
    
    train_range = 1 : floor( num_instances * 0.9 );
    test_range  = train_range( end ) + 1 : num_instances;

    train_dataset = dataset( train_range, : );
    test_dataset  = dataset( test_range, : );
    
    mlp = SimpleMLP( 'trainX', train_dataset( :, 2 : end ), ...
                     'trainY', train_dataset( :, 1 ), ...
                     'testX',  test_dataset( :, 2 : end ), ...
                     'testY',  test_dataset( :, 1 ), ...
                     'learning_rate', 0.01, ...
                     'random_seed', 12345, ...
                     'activation_fun', 'relu', ...
                     'epochs', 300);    
    trained_mlp = mlp.train();
    
    acc = trained_mlp.trainEvaluation();
    
    expected_border = 0.950;
    
    verifyTrue( testCase, acc > expected_border );
end
