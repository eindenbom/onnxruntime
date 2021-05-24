// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Holds the options for configuring a TensorRT Execution Provider instance
    /// </summary>
    public class OrtTensorRTProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        //private string _options;

        #region Constructor

        /// <summary>
        /// Constructs an empty OrtTensorRTProviderOptions instance
        /// </summary>
        public OrtTensorRTProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorRTProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Get TensorRT EP provider options
        /// </summary>
        /// <returns> return C# UTF-16 encoded string </returns>
        public string GetOptions()
        {
            var allocator = OrtAllocator.DefaultInstance;

            // Process provider options string
            IntPtr providerOptions = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorRTProviderOptions(allocator.Pointer, out providerOptions));
            using (var ortAllocation = new OrtMemoryAllocation(allocator, providerOptions, 0))
            {
                //_options = NativeOnnxValueHelper.StringFromNativeUtf8(providerOptions);
                return NativeOnnxValueHelper.StringFromNativeUtf8(providerOptions);
            }

            //return _options;
        }

        /// <summary>
        /// Updates  the configuration knobs of OrtTensorRTProviderOptions that will eventually be used to configure a TensorRT EP
        /// Please refer to the following on different key/value pairs to configure a TensorRT EP and their meaning:
        /// https://www.onnxruntime.ai/docs/reference/execution-providers/TensorRT-ExecutionProvider.html
        /// </summary>
        /// <param name="providerOptions">key/value pairs used to configure a TensorRT Execution Provider</param>
        public void UpdateOptions(Dictionary<string, string> providerOptions)
        {

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var keysArray = NativeOnnxValueHelper.ConvertNamesToUtf8(providerOptions.Keys.ToArray(), n => n, cleanupList);
                var valuesArray = NativeOnnxValueHelper.ConvertNamesToUtf8(providerOptions.Values.ToArray(), n => n, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtUpdateTensorRTProviderOptions(handle, keysArray, valuesArray, (UIntPtr)providerOptions.Count));
            }
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region Private Methods


        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtTensorRTProviderOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseTensorRTProviderOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }

    /// <summary>
    /// This helper class contains methods to handle values of provider options
    /// </summary>
    public class ProviderOptionsValueHelper
    {
        /// <summary>
        /// Parse from string and save to dictionary
        /// </summary>
        public static void StringToDict(string s, Dictionary<string, string> dict)
        {
            string[] paris = s.Split(';');

            foreach (var p in paris)
            {
                string[] keyValue = p.Split('=');
                dict.Add(keyValue[0], keyValue[1]);
            }
        }
    }

}
