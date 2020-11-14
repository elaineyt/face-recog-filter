/**
* (c) Facebook, Inc. and its affiliates. Confidential and proprietary.
*/

//==============================================================================
// Welcome to scripting in Spark AR Studio! Helpful links:
//
// Scripting Basics - https://fb.me/spark-scripting-basics
// Reactive Programming - https://fb.me/spark-reactive-programming
// Scripting Object Reference - https://fb.me/spark-scripting-reference
// Changelogs - https://fb.me/spark-changelog
//
// For projects created with v87 onwards, JavaScript is always executed in strict mode.
//==============================================================================

// Load modules
const FaceTracking = require('FaceTracking');
const Scene = require('Scene');
export const Diagnostics = require('Diagnostics');
const Time = require('Time');
const Reactive = require('Reactive');

// Store reference to a detected face
const face = FaceTracking.face(0);

// SVM
var svmjs = (function (exports) {

  /*
  This is a binary SVM and is trained using the SMO algorithm.
  Reference: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)

  Simple usage example:
  svm = svmjs.SVM();
  svm.train(data, labels);
  testlabels = svm.predict(testdata);
  */

  var SVM = function (options) {
  }

  SVM.prototype = {

    // data is NxD array of floats. labels are 1 or -1.
    train: function (data, labels, options) {

      // we need these in helper functions
      this.data = data;
      this.labels = labels;

      // parameters
      options = options || {};
      var C = options.C || 1.0; // C value. Decrease for more regularization
      var tol = options.tol || 1e-4; // numerical tolerance. Don't touch unless you're pro
      var alphatol = options.alphatol || 1e-7; // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
      var maxiter = options.maxiter || 10000; // max number of iterations
      var numpasses = options.numpasses || 10; // how many passes over data with no change before we halt? Increase for more precision.

      // instantiate kernel according to options. kernel can be given as string or as a custom function
      var kernel = linearKernel;
      this.kernelType = "linear";
      if ("kernel" in options) {
        if (typeof options.kernel === "string") {
          // kernel was specified as a string. Handle these special cases appropriately
          if (options.kernel === "linear") {
            this.kernelType = "linear";
            kernel = linearKernel;
          }
          if (options.kernel === "rbf") {
            var rbfSigma = options.rbfsigma || 0.5;
            this.rbfSigma = rbfSigma; // back this up
            this.kernelType = "rbf";
            kernel = makeRbfKernel(rbfSigma);
          }
        } else {
          // assume kernel was specified as a function. Let's just use it
          this.kernelType = "custom";
          kernel = options.kernel;
        }
      }

      // initializations
      this.kernel = kernel;
      this.N = data.length; var N = this.N;
      this.D = data[0].length; var D = this.D;
      this.alpha = zeros(N);
      this.b = 0.0;
      this.usew_ = false; // internal efficiency flag

      // Cache kernel computations to avoid expensive recomputation.
      // This could use too much memory if N is large.
      if (options.memoize) {
        this.kernelResults = new Array(N);
        for (var i = 0; i < N; i++) {
          this.kernelResults[i] = new Array(N);
          for (var j = 0; j < N; j++) {
            this.kernelResults[i][j] = kernel(data[i], data[j]);
          }
        }
      }

      // run SMO algorithm
      var iter = 0;
      var passes = 0;
      while (passes < numpasses && iter < maxiter) {

        var alphaChanged = 0;
        for (var i = 0; i < N; i++) {

          var Ei = this.marginOne(data[i]) - labels[i];
          if ((labels[i] * Ei < -tol && this.alpha[i] < C)
          || (labels[i] * Ei > tol && this.alpha[i] > 0)) {

            // alpha_i needs updating! Pick a j to update it with
            var j = i;
            while (j === i) j = randi(0, this.N);
            var Ej = this.marginOne(data[j]) - labels[j];

            // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
            var ai = this.alpha[i];
            var aj = this.alpha[j];
            var L = 0; var H = C;
            if (labels[i] === labels[j]) {
              L = Math.max(0, ai + aj - C);
              H = Math.min(C, ai + aj);
            } else {
              L = Math.max(0, aj - ai);
              H = Math.min(C, C + aj - ai);
            }

            if (Math.abs(L - H) < 1e-4) continue;

            var eta = 2 * this.kernelResult(i, j) - this.kernelResult(i, i) - this.kernelResult(j, j);
            if (eta >= 0) continue;

            // compute new alpha_j and clip it inside [0 C]x[0 C] box
            // then compute alpha_i based on it.
            var newaj = aj - labels[j] * (Ei - Ej) / eta;
            if (newaj > H) newaj = H;
            if (newaj < L) newaj = L;
            if (Math.abs(aj - newaj) < 1e-4) continue;
            this.alpha[j] = newaj;
            var newai = ai + labels[i] * labels[j] * (aj - newaj);
            this.alpha[i] = newai;

            // update the bias term
            var b1 = this.b - Ei - labels[i] * (newai - ai) * this.kernelResult(i, i)
            - labels[j] * (newaj - aj) * this.kernelResult(i, j);
            var b2 = this.b - Ej - labels[i] * (newai - ai) * this.kernelResult(i, j)
            - labels[j] * (newaj - aj) * this.kernelResult(j, j);
            this.b = 0.5 * (b1 + b2);
            if (newai > 0 && newai < C) this.b = b1;
            if (newaj > 0 && newaj < C) this.b = b2;

            alphaChanged++;

          } // end alpha_i needed updating
        } // end for i=1..N

        iter++;
        //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
        if (alphaChanged == 0) passes++;
        else passes = 0;

      } // end outer loop

      // if the user was using a linear kernel, lets also compute and store the
      // weights. This will speed up evaluations during testing time
      if (this.kernelType === "linear") {

        // compute weights and store them
        this.w = new Array(this.D);
        for (var j = 0; j < this.D; j++) {
          var s = 0.0;
          for (var i = 0; i < this.N; i++) {
            s += this.alpha[i] * labels[i] * data[i][j];
          }
          this.w[j] = s;
          this.usew_ = true;
        }
      } else {

        // okay, we need to retain all the support vectors in the training data,
        // we can't just get away with computing the weights and throwing it out

        // But! We only need to store the support vectors for evaluation of testing
        // instances. So filter here based on this.alpha[i]. The training data
        // for which this.alpha[i] = 0 is irrelevant for future.
        var newdata = [];
        var newlabels = [];
        var newalpha = [];
        for (var i = 0; i < this.N; i++) {
          //console.log("alpha=%f", this.alpha[i]);
          if (this.alpha[i] > alphatol) {
            newdata.push(this.data[i]);
            newlabels.push(this.labels[i]);
            newalpha.push(this.alpha[i]);
          }
        }

        // store data and labels
        this.data = newdata;
        this.labels = newlabels;
        this.alpha = newalpha;
        this.N = this.data.length;
        //console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
      }

      var trainstats = {};
      trainstats.iters = iter;
      return trainstats;
    },

    // inst is an array of length D. Returns margin of given example
    // this is the core prediction function. All others are for convenience mostly
    // and end up calling this one somehow.
    marginOne: function (inst) {

      var f = this.b;
      // if the linear kernel was used and w was computed and stored,
      // (i.e. the svm has fully finished training)
      // the internal class variable usew_ will be set to true.
      if (this.usew_) {

        // we can speed this up a lot by using the computed weights
        // we computed these during train(). This is significantly faster
        // than the version below
        for (var j = 0; j < this.D; j++) {
          f += inst[j] * this.w[j];
        }

      } else {

        for (var i = 0; i < this.N; i++) {
          f += this.alpha[i] * this.labels[i] * this.kernel(inst, this.data[i]);
        }
      }

      return f;
    },

    predictOne: function (inst) {
      return this.marginOne(inst) > 0 ? 1 : -1;
    },

    // data is an NxD array. Returns array of margins.
    margins: function (data) {

      // go over support vectors and accumulate the prediction.
      var N = data.length;
      var margins = new Array(N);
      for (var i = 0; i < N; i++) {
        margins[i] = this.marginOne(data[i]);
      }
      return margins;

    },

    kernelResult: function (i, j) {
      if (this.kernelResults) {
        return this.kernelResults[i][j];
      }
      return this.kernel(this.data[i], this.data[j]);
    },

    // data is NxD array. Returns array of 1 or -1, predictions
    predict: function (data) {
      var margs = this.margins(data);
      for (var i = 0; i < margs.length; i++) {
        margs[i] = margs[i] > 0 ? 1 : -1;
      }
      return margs;
    },

    // THIS FUNCTION IS NOW DEPRECATED. WORKS FINE BUT NO NEED TO USE ANYMORE.
    // LEAVING IT HERE JUST FOR BACKWARDS COMPATIBILITY FOR A WHILE.
    // if we trained a linear svm, it is possible to calculate just the weights and the offset
    // prediction is then yhat = sign(X * w + b)
    getWeights: function () {

      // DEPRECATED
      var w = new Array(this.D);
      for (var j = 0; j < this.D; j++) {
        var s = 0.0;
        for (var i = 0; i < this.N; i++) {
          s += this.alpha[i] * this.labels[i] * this.data[i][j];
        }
        w[j] = s;
      }
      return { w: w, b: this.b };
    },

    toJSON: function () {

      if (this.kernelType === "custom") {
        console.log("Can't save this SVM because it's using custom, unsupported kernel...");
        return {};
      }

      json = {}
      json.N = this.N;
      json.D = this.D;
      json.b = this.b;

      json.kernelType = this.kernelType;
      if (this.kernelType === "linear") {
        // just back up the weights
        json.w = this.w;
      }
      if (this.kernelType === "rbf") {
        // we need to store the support vectors and the sigma
        json.rbfSigma = this.rbfSigma;
        json.data = this.data;
        json.labels = this.labels;
        json.alpha = this.alpha;
      }

      return json;
    },

    fromJSON: function (json) {

      this.N = json.N;
      this.D = json.D;
      this.b = json.b;

      this.kernelType = json.kernelType;
      if (this.kernelType === "linear") {

        // load the weights!
        this.w = json.w;
        this.usew_ = true;
        this.kernel = linearKernel; // this shouldn't be necessary
      }
      else if (this.kernelType == "rbf") {

        // initialize the kernel
        this.rbfSigma = json.rbfSigma;
        this.kernel = makeRbfKernel(this.rbfSigma);

        // load the support vectors
        this.data = json.data;
        this.labels = json.labels;
        this.alpha = json.alpha;
      } else {
        console.log("ERROR! unrecognized kernel type." + this.kernelType);
      }
    }
  }

  // Kernels
  function makeRbfKernel(sigma) {
    return function (v1, v2) {
      var s = 0;
      for (var q = 0; q < v1.length; q++) { s += (v1[q] - v2[q]) * (v1[q] - v2[q]); }
      return Math.exp(-s / (2.0 * sigma * sigma));
    }
  }

  function linearKernel(v1, v2) {
    var s = 0;
    for (var q = 0; q < v1.length; q++) { s += v1[q] * v2[q]; }
    return s;
  }

  // Misc utility functions
  // generate random floating point number between a and b
  function randf(a, b) {
    return Math.random() * (b - a) + a;
  }

  // generate random integer between a and b (b excluded)
  function randi(a, b) {
    return Math.floor(Math.random() * (b - a) + a);
  }

  // create vector of zeros of length n
  function zeros(n) {
    var arr = new Array(n);
    for (var i = 0; i < n; i++) { arr[i] = 0; }
    return arr;
  }

  // export public members
  exports = exports || {};
  exports.SVM = SVM;
  exports.makeRbfKernel = makeRbfKernel;
  exports.linearKernel = linearKernel;
  return exports;

})(typeof module != 'undefined' && module.exports);  // add exports to module.exports if in node.js

let faceMap = new Map();
faceMap.set("Elaine", [0.04722372815012932,0.03462549299001694,0.04461556300520897,0.039947863668203354,0.10853677988052368,0.13221219182014465,0.1002160906791687,0.10850419849157333,0.08514878153800964,0.10348457098007202,0.07470040023326874,0.080085888504982,0.056285515427589417,0.04516131803393364,0.03571208566427231,0.040829386562108994,0.10272730886936188,0.1073479950428009,0.10758905112743378,0.13390301167964935,0.07434972375631332,0.07877238094806671,0.08663585036993027,0.10578492283821106,0.05628559738397598,0.01648644544184208,0.008965671993792057,0.0766313299536705,0.10402537882328033,0.07198775559663773,0.09537604451179504,0.05570996180176735,0.07594140619039536,0.0511748306453228,0.06755340099334717,0.08144526183605194,0.008934247307479382,0.07331757992506027,0.09328026324510574,0.07518277317285538,0.10587651282548904,0.049681857228279114,0.06454116106033325,0.057085320353507996,0.07878442108631134,0.08176601678133011,0.07537273317575455,0.09986387193202972,0.07406550645828247,0.10191119462251663,0.05387074127793312,0.07152043282985687,0.05536831170320511,0.07456714659929276,0.08138469606637955,0.05458546057343483,0.02941354177892208,0.08314429223537445,0.03100004233419895,0.04714221507310867,0.04585115239024162,0.07312759011983871,0.1537831425666809,0.08150018006563187,0.12842224538326263,0.05205780640244484,0.02927463874220848,0.08724869042634964,0.11636623740196228,0.16313816606998444,0.05516454204916954,0.044157400727272034,0.07009432464838028,0.028665445744991302,0.047531843185424805,0.15189750492572784,0.08822855353355408,0.11591142416000366,0.052405718713998795,0.028796203434467316,0.16391970217227936,0.029712535440921783,0.03870861232280731,0.06958737969398499,0.1264398694038391,0.06809017807245255,0.0984763577580452,0.13451561331748962,0.031189020723104477,0.12712694704532623,0.13548806309700012]);
faceMap.set("Alice", [0.04800700023770332,0.03245558589696884,0.042281657457351685,0.037735871970653534,0.10432510823011398,0.1281566470861435,0.09342973679304123,0.1006942167878151,0.08489950746297836,0.10396675020456314,0.07350214570760727,0.08024390786886215,0.05221979692578316,0.04234690219163895,0.03301917761564255,0.037970494478940964,0.09542033821344376,0.10530199110507965,0.1044863611459732,0.12660712003707886,0.0744347870349884,0.0811736062169075,0.08579700440168381,0.10507768392562866,0.05173107236623764,0.014975612051784992,0.008351298049092293,0.07367681711912155,0.10039079189300537,0.06872967630624771,0.08948574960231781,0.05724421888589859,0.07702550292015076,0.05247077718377113,0.06873811036348343,0.07348674535751343,0.008239560760557652,0.06941520422697067,0.09145967662334442,0.07316050678491592,0.09899718314409256,0.05218314751982689,0.06769273430109024,0.057712990790605545,0.07828006893396378,0.07371198385953903,0.07189469039440155,0.0972062349319458,0.0713275745511055,0.09560347348451614,0.05611136183142662,0.07380545139312744,0.05652898922562599,0.07504797726869583,0.0733034759759903,0.04517730697989464,0.04135851562023163,0.0842687115073204,0.027910461649298668,0.03883031755685806,0.04841131344437599,0.07387862354516983,0.14242389798164368,0.08249510824680328,0.11714870482683182,0.04653952270746231,0.02475138008594513,0.08069667220115662,0.10630952566862106,0.15602834522724152,0.04665955901145935,0.04796707257628441,0.07235394418239594,0.02723263017833233,0.03977538272738457,0.1413845717906952,0.08125872910022736,0.10633661597967148,0.04588859900832176,0.02169043757021427,0.1520656943321228,0.02740471251308918,0.03777582198381424,0.06571422517299652,0.1211109459400177,0.06470585614442825,0.09193500131368637,0.1315213143825531,0.028436951339244843,0.1210031658411026,0.13115842640399933]);
faceMap.set("Brandon",
[0.0523444265127182,0.033863067626953125,0.04453789442777634,0.03934778273105621,0.10615673661231995,0.1311827152967453,0.10032017529010773,0.10551325976848602,0.09160604327917099,0.10979460179805756,0.07812519371509552,0.08357736468315125,0.05302806943655014,0.04393935576081276,0.0336848609149456,0.03898271545767784,0.09827366471290588,0.10310472548007965,0.10901442915201187,0.13191190361976624,0.07795616239309311,0.08345921337604523,0.09021098166704178,0.10815972089767456,0.05319149047136307,0.015599314123392105,0.0083809494972229,0.07448054850101471,0.10092940181493759,0.07384385168552399,0.09286246448755264,0.061058495193719864,0.07978396862745285,0.05496402084827423,0.06948933750391006,0.07377063482999802,0.008396659977734089,0.07111646980047226,0.09068614989519119,0.07736369222402573,0.10241692513227463,0.055126748979091644,0.06964317709207535,0.06023463234305382,0.07896499335765839,0.07403266429901123,0.07298837602138519,0.0967055931687355,0.07580065727233887,0.09860961139202118,0.05894128605723381,0.07567665725946426,0.058473262935876846,0.07526661455631256,0.07353167235851288,0.05135893449187279,0.03286051005125046,0.07903025299310684,0.024845240637660027,0.041489750146865845,0.042692236602306366,0.06614808738231659,0.14464488625526428,0.08259010314941406,0.12282541394233704,0.04494088143110275,0.02139260061085224,0.08438291400671005,0.10894341766834259,0.1556454300880432,0.04865942895412445,0.047091081738471985,0.07103873789310455,0.02644195407629013,0.039458680897951126,0.14708279073238373,0.0852837860584259,0.11030443012714386,0.046576082706451416,0.023771628737449646,0.15750688314437866,0.02689911611378193,0.04155866056680679,0.0675884336233139,0.12595883011817932,0.06814634799957275,0.09369765967130661,0.13545112311840057,0.026479575783014297,0.1254088282585144,0.13490499556064606]);
faceMap.set("Christine",
[0.04800700023770332,0.03245558589696884,0.042281657457351685,0.037735871970653534,0.10432510823011398,0.1281566470861435,0.09342973679304123,0.1006942167878151,0.08489950746297836,0.10396675020456314,0.07350214570760727,0.08024390786886215,0.05221979692578316,0.04234690219163895,0.03301917761564255,0.037970494478940964,0.09542033821344376,0.10530199110507965,0.1044863611459732,0.12660712003707886,0.0744347870349884,0.0811736062169075,0.08579700440168381,0.10507768392562866,0.05173107236623764,0.014975612051784992,0.008351298049092293,0.07367681711912155,0.10039079189300537,0.06872967630624771,0.08948574960231781,0.05724421888589859,0.07702550292015076,0.05247077718377113,0.06873811036348343,0.07348674535751343,0.008239560760557652,0.06941520422697067,0.09145967662334442,0.07316050678491592,0.09899718314409256,0.05218314751982689,0.06769273430109024,0.057712990790605545,0.07828006893396378,0.07371198385953903,0.07189469039440155,0.0972062349319458,0.0713275745511055,0.09560347348451614,0.05611136183142662,0.07380545139312744,0.05652898922562599,0.07504797726869583,0.0733034759759903,0.04517730697989464,0.04135851562023163,0.0842687115073204,0.027910461649298668,0.03883031755685806,0.04841131344437599,0.07387862354516983,0.14242389798164368,0.08249510824680328,0.11714870482683182,0.04653952270746231,0.02475138008594513,0.08069667220115662,0.10630952566862106,0.15602834522724152,0.04665955901145935,0.04796707257628441,0.07235394418239594,0.02723263017833233,0.03977538272738457,0.1413845717906952,0.08125872910022736,0.10633661597967148,0.04588859900832176,0.02169043757021427,0.1520656943321228,0.02740471251308918,0.03777582198381424,0.06571422517299652,0.1211109459400177,0.06470585614442825,0.09193500131368637,0.1315213143825531,0.028436951339244843,0.1210031658411026,0.13115842640399933]);

var faceData = new Array(91);

function compareTwoFaces(person1, person2) {
  // Get face data for person1 and person2
  var data = [faceMap.get(person1),faceMap.get(person2)]
  var labels = [-1, 1]

  // Train SVM
  var svm = new svmjs.SVM();
  svm.train(data, labels, { kernel: svmjs.makeRbfKernel(0.5), C: 1.0 });

  // Make prediction
  //Diagnostics.log(faceData)
  var prediction = svm.predict([faceData]);
  if(prediction[0] == -1){
    return person1;
  }
  return person2;
}

function findMostSimilarFace() {
  // call this function in time interval, get face data values
    var moreSim12 = compareTwoFaces("Elaine", "Alice");
    var moreSim34 = compareTwoFaces("Brandon", "Christine");
  // moreSim5-6 = compareTwoFaces(person5, person6, facedata)
  // moreSim7-8 = compareTwoFaces(person7, person8, facedata)
  // moreSim9-10 = compareTwoFaces(person9, person10, facedata)
  // moreSim11-12 = compareTwoFaces(person11, person12, facedata)
  // moreSim13-14 = compareTwoFaces(person13, person14, facedata)
  // moreSim15-16 = compareTwoFaces(person15, person16, facedata)

    var mostSim14 = compareTwoFaces(moreSim12, moreSim34);
  // mostSim5-8 = compareTwoFaces(moreSim5-6, moreSim7-8, facedata)
  // mostSim9-12 = compareTwoFaces(moreSim9-10, moreSim11-12, facedata)
  // mostSim13-16 = compareTwoFaces(moreSim13-14, moreSim15-16, facedata)

  // mostSim1-8 = compareTwoFaces(mostSim1-4, mostSim5-8, facedata)
  // mostSim9-16 = compareTwoFaces(mostSim9-12, mostSim13-16, facedata)

  // mostSim1-16 = compareTwoFaces(mostSim1-8, mostSim9-16, facedata)

  return mostSim14;
}

function run() {
  var facialPoints = [face.mouth.leftCorner,
    face.mouth.rightCorner,
    face.nose.leftNostril,
    face.nose.rightNostril,
    face.nose.tip,
    face.rightEyebrow.insideEnd,
    face.rightEyebrow.outsideEnd,
    face.leftEyebrow.insideEnd,
    face.leftEyebrow.outsideEnd,
    face.rightEye.insideCorner,
    face.rightEye.outsideCorner,
    face.leftEye.insideCorner,
    face.leftEye.outsideCorner,
    face.chin.tip]

    var index = 0;
    for(var i = 0; i < facialPoints.length; i++){
      for(var j = i+1; j < facialPoints.length; j++){
        faceData[index] = Reactive.distance(facialPoints[i],facialPoints[j]).pinLastValue();
        index++;
      }
    }

    Diagnostics.log(findMostSimilarFace());
  }

  function printFaceValues(){
    var facialPoints = [face.mouth.leftCorner,
      face.mouth.rightCorner,
      face.nose.leftNostril,
      face.nose.rightNostril,
      face.nose.tip,
      face.rightEyebrow.insideEnd,
      face.rightEyebrow.outsideEnd,
      face.leftEyebrow.insideEnd,
      face.leftEyebrow.outsideEnd,
      face.rightEye.insideCorner,
      face.rightEye.outsideCorner,
      face.leftEye.insideCorner,
      face.leftEye.outsideCorner,
      face.chin.tip]
      var facialDistances = new Array(91);

      var index = 0;
      for(var i = 0; i < facialPoints.length; i++){
        for(var j = i+1; j < facialPoints.length; j++){
          facialDistances[index] = Reactive.distance(facialPoints[i],facialPoints[j]).pinLastValue();
          index++;
        }
      }

      Diagnostics.log(facialDistances);



      /*
      Diagnostics.log("Face Data:")
      Diagnostics.log("[" + Reactive.distance(face.leftEye.center, face.rightEye.center).pinLastValue() + "," +
      Reactive.distance(face.leftEye.center, face.nose.tip).pinLastValue() + "," +
      Reactive.distance(face.nose.tip, face.rightEye.center).pinLastValue() + "," +
      Reactive.distance(face.chin.tip, face.forehead.top).pinLastValue() + "," +
      Reactive.distance(face.chin.tip, face.rightEye.outsideCorner).pinLastValue() + "," +
      Reactive.distance(face.chin.tip, face.leftEye.outsideCorner).pinLastValue() + "," +
      Reactive.distance(face.mouth.leftCorner, face.mouth.rightCorner).pinLastValue() + "," +
      Reactive.distance(face.leftEye.insideCorner, face.mouth.leftCorner).pinLastValue() + "," +
      Reactive.distance(face.rightEye.insideCorner, face.mouth.rightCorner).pinLastValue() + "]");*/
    }
    const intervalTimer = Time.setInterval(run, 2000);
    //const intervalTimer = Time.setInterval(printFaceValues, 3000);
