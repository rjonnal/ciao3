from acedev5 import *
 
class PyAcedev5:
    """Python class for handling acedev5 wrapper."""
    
    dmId = -1
    values = []
    nAct = 0
    
    def __init__(self, serial):
        """Constructor."""
        result, dmId  = acedev5Init( serial )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("Constructor: " + msg)
        self.dmId = dmId
        self.nbAct = self.GetNbActuator()
        #self.values = dblArray( self.nbAct )
        self.values = [0] * self.nbAct
        
    def __del__(self):
        """Destructor."""
        result = acedev5Release( self.dmId )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("Destructor: " + msg)
    
    def GetNbActuator(self):
        """Get number of actuators."""
        result, nbAct = acedev5GetNbActuator( self.dmId )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("GetNbActuator(): " + msg)
        return nbAct
        
    def GetOffset(self):
        """Get offsets matrix."""
        offsets = dblArray( self.nbAct )
        result = acedev5GetOffset( self.dmId, offsets )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("GetOffset(): " + msg)
        finalOff = []
        for i in range( self.nbAct ):
            finalOff.append( offsets[i] )
        return finalOff

    # Send Intensity
    def Send(self):
        """Send values to DM."""
        finaleValues = dblArray( self.nbAct )
        for i in range( min( self.nbAct, len( self.values) ) ):
            finaleValues[i] = self.values[i]
        result = acedev5Send( self.dmId, finaleValues )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("Send(): " + msg)
        
    def DACReset(self):
        """Reset DAC (Digital to Analog Converter) values."""
        result = acedev5SoftwareDACReset( self.dmId )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("DACReset(): " + msg)

    # Send Pattern        
    def StartPattern(self, patterns, nRepeat):
        """Start pattern."""
        nPattern = len(patterns) / self.nbAct
        finalPatterns = dblArray( self.nbAct * nPattern )
        for i in range ( nPattern * self.nbAct ):
            finalPatterns[i] = patterns[i]
        result = acedev5StartPattern( self.dmId, finalPatterns, nPattern, nRepeat )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("StartPattern(): " + msg)
        
    def StopPattern(self):
        """Stop pattern."""
        result = acedev5StopPattern( self.dmId )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("StopPattern(): " + msg)
        
    def QueryPattern(self):
        """Query pattern status."""
        result, status = acedev5QueryPattern( self.dmId )
        if not result == acecsSUCCESS:
            id, msg = self.ErrGetStatus()
            raise Exception("QueryPattern(): " + msg)
        return status

    # Error handling
    def ErrDisplay(self):
        """Display acedev5 errors."""
        acecsErrDisplay()
        
    def ErrGetStatus(self):
        """Get last error."""
        result, errorId, errMsg = acecsErrGetStatus()
        if not result == acecsSUCCESS:
            self.ErrDisplay()
            raise Exception("ErrGetStatus(): Can't get error message")
        return errorId, errMsg