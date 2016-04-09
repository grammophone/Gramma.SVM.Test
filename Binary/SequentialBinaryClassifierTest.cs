using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Grammophone.Vectors;
using Grammophone.Kernels;
using Grammophone.Optimization;
using Grammophone.SVM;
using Grammophone.SVM.CoordinateDescent;

namespace SvmTest.Binary
{
	[TestClass]
	public class SerialCoordinateDescentBinaryClassifierTest : ClassifierTest
	{
		#region Protected methods

		protected override BinaryClassifier<Vector> CreateClassifier(Kernel<Vector> kernel)
		{
			return new SerialCoordinateDescentBinaryClassifier<Vector>(kernel);
		}

		#endregion
	}
}
