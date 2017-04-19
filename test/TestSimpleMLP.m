function tests = TestSimpleMLP
    tests = functiontests( localfunctions );
end

function setupOnce( testCase )
    fprintf( '\nNow setuping test ...\n' );

    addpath( '../' ); % Add @SimpleMLP

    dataset = load( '../datasets/iris_scaled.csv' );

    %% Add sample train & test datasets.
    num_instances = size( dataset, 1 );

    train_range = 1 : floor( num_instances * 0.9 );
    test_range  = train_range( end ) + 1 : num_instances;

    train_dataset = dataset( train_range, : );
    test_dataset  = dataset( test_range, : );

    testCase.TestData.train_dataset = train_dataset;
    testCase.TestData.test_dataset  = test_dataset;

    %% Add sample trained mlp model with default parameter.
    %% But only random_seed is set for test. 
    mlp = SimpleMLP( 'trainX',      train_dataset( :, 2 : end ), ...
                     'trainY',      train_dataset( :, 1 ), ...
                     'testX',       test_dataset( :, 2 : end ), ...
                     'testY',       test_dataset( :, 1 ), ...
                     'random_seed', 10000 );
    testCase.TestData.mlp = mlp.train();
end

function initialParameterTest( testCase )
    fprintf( '\nStart initialParameterTest\n' );

    mlp = testCase.TestData.mlp;

    % About the way how to set parameters, please see 'testOtherModelTest'
    verifyEqual( testCase, mlp.learning_rate, 0.01 );
    verifyEqual( testCase, mlp.count_hneuron, 5 );
    verifyEqual( testCase, mlp.epochs, 500 );
    verifyEqual( testCase, mlp.output_activation_fun, 'sig' );
    verifyEqual( testCase, mlp.activation_fun, 'sig' );
end

function predictionTest( testCase )
    fprintf( '\nStart predictionTest\n' );

    mlp = testCase.TestData.mlp;
    test_dataset = testCase.TestData.test_dataset;

    expected_label        = test_dataset( 1, 1 );
    [ val actual_output ] = max( mlp.predict( test_dataset( 1, 2 : end ) ) );

    verifyEqual( testCase, expected_label, actual_output );
end

function testOtherModelTest( testCase )
    fprintf( '\nStart testOtherModelTest\n' );

    train_dataset = testCase.TestData.train_dataset;
    test_dataset  = testCase.TestData.test_dataset;

    mlp = UsefulMLP( 'trainX', train_dataset( :, 2 : end ), ...
                     'trainY', train_dataset( :, 1 ), ...
                     'testX',  test_dataset( :, 2 : end ), ...
                     'testY',  test_dataset( :, 1 ), ...
                     'learning_rate', 0.01, ...
                     'count_hneuron', [3 3], ... % 2 hidden layers
                     'random_seed', 12345, ...
                     'activation_fun', 'relu', ...
                     'output_activation_fun', 'sig', ...
                     'epochs', 300 );

    trained_mlp = mlp.train();

    train_acc = trained_mlp.trainEvaluation();
    expected_border = 0.980;
    verifyTrue( testCase, train_acc > expected_border );

    test_acc = trained_mlp.testEvaluation();
    expected_border = 0.980;
    verifyTrue( testCase, test_acc > expected_border );
end
