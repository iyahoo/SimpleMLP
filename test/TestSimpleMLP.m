function tests = TestSimpleMLP
    tests = functiontests( localfunctions );
end

function testUnion( testCase )
% Requier Statistic and Machine Learning Toolbox
    addpath( '../' );

    % First column is class label, others are features.
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
                     'count_hneuron', [3 3], ... % 2 hidden layers
                     'random_seed', 12345, ...
                     'activation_fun', 'relu', ...
                     'output_activation_fun', 'sig', ...
                     'epochs', 300 );

    % You have to call train() yourself
    % Do not forget to assign trained object.
    trained_mlp = mlp.train();

    train_acc = trained_mlp.trainEvaluation();
    disp( train_acc );
    expected_border = 0.980;
    verifyTrue( testCase, train_acc > expected_border );

    test_acc = trained_mlp.testEvaluation();
    disp( test_acc );
    expected_border = 0.980;
    verifyTrue( testCase, test_acc > expected_border );
end

function initialParameterTest( testCase )
    addpath( '../' );

    dataset = load( '../datasets/iris_scaled.csv' );

    num_instances = size( dataset, 1 );

    train_range = 1 : floor( num_instances * 0.9 );
    test_range  = train_range( end ) + 1 : num_instances;

    train_dataset = dataset( train_range, : );
    test_dataset  = dataset( test_range, : );

    mlp = SimpleMLP( 'trainX', train_dataset( :, 2 : end ), ...
                     'trainY', train_dataset( :, 1 ), ...
                     'testX',  test_dataset( :, 2 : end ), ...
                     'testY',  test_dataset( :, 1 ));

    % You can see default parameters in SimpleMLP.m Line 40 - 47
    verifyEqual( testCase, mlp.learning_rate, 0.01 );
    verifyEqual( testCase, mlp.count_hneuron, 5 );
    verifyEqual( testCase, mlp.epochs, 500 );
end

function predictionTest( testCase )
    addpath( '../' );

    dataset = load( '../datasets/iris_scaled.csv' );

    num_instances = size( dataset, 1 );

    train_range = 1 : floor( num_instances * 0.9 );
    test_range  = train_range( end ) + 1 : num_instances;

    train_dataset = dataset( train_range, : );
    test_dataset  = dataset( test_range, : );

    mlp = SimpleMLP( 'trainX', train_dataset( :, 2 : end ), ...
                     'trainY', train_dataset( :, 1 ), ...
                     'testX',  test_dataset( :, 2 : end ), ...
                     'testY',  test_dataset( :, 1 ));

    mlp = mlp.train();

    expected_label        = test_dataset( 1, 1 );
    [ val actual_output ] = max( mlp.predict( test_dataset( 1, 2 : end ) ) );

    verifyEqual( testCase, expected_label, actual_output );
end
